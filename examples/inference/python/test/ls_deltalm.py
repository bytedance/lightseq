import time
import argparse

import torch
import lightseq.inference as lsi
from modeling_deltalm import DeltaLMForConditionalGeneration
from transformers import XLMRobertaTokenizer

def ls_deltalm(model, inputs):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    generated_ids = model.infer(inputs)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return generated_ids, end_time - start_time


def hf_deltalm(model, inputs):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        generated_ids = model.generate(inputs.to("cuda:0"), max_length=50)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return generated_ids, end_time - start_time


def ls_generate(model, tokenizer, inputs_id):
    print("=========lightseq=========")
    print("lightseq generating...")
    ls_res_ids, ls_time = ls_deltalm(model, inputs_id)
    ls_res_ids = [ids[0] for ids in ls_res_ids[0]]
    for i in range(len(ls_res_ids)):
        ls_res_ids[i] = [idx for idx in ls_res_ids[i] if idx < len(tokenizer)]
    ls_res = tokenizer.batch_decode(ls_res_ids, skip_special_tokens=True)
    print(f"lightseq time: {ls_time}s")
    print("lightseq results:")
    for sent in ls_res:
        print(sent)
    return ls_res

def hf_generate(model, tokenizer, inputs_id):
    print("=========huggingface=========")
    print("huggingface generating...")
    hf_res_ids, hf_time = hf_deltalm(model, inputs_id)
    hf_res_ids = hf_res_ids.cpu().numpy().tolist()
    for i in range(len(hf_res_ids)):
        hf_res_ids[i] = [idx for idx in hf_res_ids[i] if idx < len(tokenizer)]
    hf_res = tokenizer.batch_decode(hf_res_ids, skip_special_tokens=True)
    print(f"huggingface time: {hf_time}s")
    print("huggingface results:")
    for sent in hf_res:
        print(sent)
    return hf_res


def warmup(tokenizer, ls_model, hf_model, sentences):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True)
    inputs_id = inputs["input_ids"]

    ls_generate(ls_model, tokenizer, inputs_id)
    hf_generate(hf_model, tokenizer, inputs_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_input", action="store_true")
    args = parser.parse_args()

    print("initializing xlmroberta tokenizer...")
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    print("creating lightseq model...")
    ls_model = lsi.Deltalm("../export/lightseq_deltalm_converted_corrected.hdf5", 128)

    print("creating huggingface model...")
    hf_model = DeltaLMForConditionalGeneration.from_pretrained('../export/deltalm_converted_corrected/')
    hf_model.to("cuda:0")

    sentences = [
        "I love that girl, but <mask> does not <mask> me.",
        "She is so <mask> that I can not help glance at <mask>.",
        "Nothing's gonna <mask> my love for you.",
        "Drop everything now. Meet me in the pouring <mask>. Kiss me on the sidewalk.",
    ]


    print("====================START warmup====================")
    warmup(tokenizer, ls_model, hf_model, sentences)
    print("====================END warmup====================")

    while True:
        if args.user_input:
            sentences = [input("input the masked sentence:\n")]

        print("tokenizing the sentences...")
        inputs = tokenizer(sentences, return_tensors="pt", padding=True)
        inputs_id = inputs["input_ids"]

        res1 = ls_generate(ls_model, tokenizer, inputs_id)
        res2 = hf_generate(hf_model, tokenizer, inputs_id)
        print('Comparing side by side')
        for item_res1, item_res2 in zip(res1, res2):
            min_length = min(len(item_res1), len(item_res2))
            print('-'*50)
            print(item_res1[:min_length])
            print('->')
            print(item_res2[:min_length])
            assert item_res1[:min_length] == item_res2[:min_length]

        if not args.user_input:
            break


if __name__ == "__main__":
    main()
