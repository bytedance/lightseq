import time
import argparse
import torch
import lightseq.inference as lsi
from transformers import T5Tokenizer, T5ForConditionalGeneration
import sentencepiece

def ls_t5(model, inputs):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    generated_ids = model.infer(inputs)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return generated_ids, end_time - start_time


def hf_t5(model, inputs):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    generated_ids = model.generate(inputs.to("cuda:0"), max_length=50)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return generated_ids, end_time - start_time


def ls_generate(model, tokenizer, inputs_id):
    print("=========lightseq=========")
    print("lightseq generating...")
    ls_res_ids, ls_time = ls_t5(model, inputs_id)
    print(ls_res_ids)
    ls_res_ids = [ids[0] for ids in ls_res_ids[0]]
    ls_res = tokenizer.batch_decode(ls_res_ids, skip_special_tokens=False)
    print(f"lightseq time: {ls_time}s")
    print("lightseq results:")
    for sent in ls_res:
        print(sent)


def hf_generate(model, tokenizer, inputs_id):
    print("=========huggingface=========")
    print("huggingface generating...")
    hf_res_ids, hf_time = hf_t5(model, inputs_id)
    hf_res = tokenizer.batch_decode(hf_res_ids)
    print(f"huggingface time: {hf_time}s")
    print("huggingface results:")
    for sent in hf_res:
        print(sent)


def warmup(tokenizer, ls_model, hf_model, sentences):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True)
    inputs_id = inputs["input_ids"]

    ls_generate(ls_model, tokenizer, inputs_id)
    # hf_generate(hf_model, tokenizer, inputs_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_input", action="store_true")
    args = parser.parse_args()

    print("initializing t5 tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    print("creating lightseq model...")
    ls_model = lsi.T5("lightseq_t5_small.hdf5", 128)
    # ls_model = None

    # print("creating huggingface model...")
    # hf_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    # hf_model.to("cuda:0")
    # hf_model.eval()
    hf_model = None

    # sentences = [
    #     'The <extra_id_0> walks in <extra_id_1> park'
    #     # 'summerize: Tom and Alice go to cinema, and watched the most impactful movie'
    # ]


    sentences = [
        'The <extra_id_0> walks in <extra_id_1> park',
        'summerize: Tom and Alice go to cinema, and watched the most impactful movie',
        "She is so <mask> that I can not help glance at <mask>.",
        "Nothing's gonna <mask> my love for you.",
    ]

    print("====================START warmup====================")
    warmup(tokenizer, ls_model, hf_model, sentences)
    print("====================END warmup====================")
    exit()
    while True:
        if args.user_input:
            sentences = [input("input the masked sentence:\n")]

        print("tokenizing the sentences...")
        inputs = tokenizer(sentences, return_tensors="pt", padding=True)
        inputs_id = inputs["input_ids"]
        
        # ls_generate(ls_model, tokenizer, inputs_id)
        hf_generate(hf_model, tokenizer, inputs_id)

        if not args.user_input:
            break


if __name__ == "__main__":
    main()
