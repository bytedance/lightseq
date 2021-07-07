import time
import argparse

import torch
import lightseq.inference as lsi
from transformers import FSMTForConditionalGeneration, FSMTTokenizer


def ls_fsmt(model, inputs):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    generated_ids = model.infer(inputs)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return generated_ids, end_time - start_time


def hf_fsmt(model, inputs):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    generated_ids = model.generate(inputs.to("cuda:0"), max_length=50, num_beams=4)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return generated_ids, end_time - start_time


def ls_generate(model, tokenizer, inputs_id):
    print("=========lightseq=========")
    print("lightseq generating...")
    ls_res_ids, ls_time = ls_fsmt(model, inputs_id)
    ls_res_ids = [ids[0] for ids in ls_res_ids[0]]
    ls_res = tokenizer.batch_decode(ls_res_ids, skip_special_tokens=True)
    print(f"lightseq time: {ls_time}s")
    print("lightseq results:")
    for sent in ls_res:
        print(sent)


def hf_generate(model, tokenizer, inputs_id):
    print("=========huggingface=========")
    print("huggingface generating...")
    hf_res_ids, hf_time = hf_fsmt(model, inputs_id)
    hf_res = tokenizer.batch_decode(hf_res_ids, skip_special_tokens=True)
    print(f"huggingface time: {hf_time}s")
    print("huggingface results:")
    for sent in hf_res:
        print(sent)


def warmup(tokenizer, ls_model, hf_model, sentences):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True)
    inputs_id = inputs["input_ids"]

    ls_generate(ls_model, tokenizer, inputs_id)
    hf_generate(hf_model, tokenizer, inputs_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_input", action="store_true")
    args = parser.parse_args()

    print("initializing fsmt tokenizer...")
    tokenizer = FSMTTokenizer.from_pretrained("facebook/wmt19-en-de")

    print("creating lightseq model...")
    ls_model = lsi.Transformer("lightseq_fsmt_wmt19ende.hdf5", 50)
    print("creating huggingface model...")
    hf_model = FSMTForConditionalGeneration.from_pretrained("facebook/wmt19-en-de")
    hf_model.to("cuda:0")

    sentences = [
        "Machine learning is great, isn't it?",
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

        ls_generate(ls_model, tokenizer, inputs_id)
        hf_generate(hf_model, tokenizer, inputs_id)

        if not args.user_input:
            break


if __name__ == "__main__":
    main()
