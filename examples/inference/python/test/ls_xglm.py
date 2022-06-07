import time
import argparse
import torch
import numpy as np
import lightseq.inference as lsi

from transformers import AutoTokenizer, AutoModelForCausalLM


def ls_xglm(model, inputs):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    generated_ids = model.sample(inputs)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return generated_ids, end_time - start_time


def hf_xglm(model, inputs, tokenizer):
    inputs = inputs.to("cuda:0")
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    generated_ids = model.generate(inputs, max_length=100, top_k=1)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return generated_ids, end_time - start_time


def ls_generate(model, tokenizer, inputs):
    print("=========lightseq=========")
    print("lightseq generating...")
    ls_res_ids, ls_time = ls_xglm(model, inputs)
    ls_res = tokenizer.batch_decode(ls_res_ids, skip_special_tokens=True)
    print(f"lightseq time: {ls_time}s")
    print("lightseq results:")
    for sent in ls_res:
        print(sent)


def hf_generate(model, tokenizer, inputs):
    print("=========huggingface=========")
    print("huggingface generating...")
    hf_res_ids, hf_time = hf_xglm(model, inputs, tokenizer)
    hf_res = tokenizer.batch_decode(hf_res_ids, skip_special_tokens=True)
    print(f"huggingface time: {hf_time}s")
    print("huggingface results:")
    for sent in hf_res:
        print(sent)


def warmup(ls_tokenizer, hf_tokenizer, ls_model, hf_model, sentences):
    ls_inputs = ls_tokenizer(sentences, return_tensors="pt")["input_ids"]
    hf_inputs = hf_tokenizer(sentences, return_tensors="pt")["input_ids"]

    ls_generate(ls_model, ls_tokenizer, ls_inputs)
    hf_generate(hf_model, hf_tokenizer, hf_inputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_input", action="store_true")
    args = parser.parse_args()

    print("initializing xglm tokenizer...")

    ls_tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-1B")
    # lightseq use len(tokenizer) as pad_token in default
    ls_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    hf_tokenizer = AutoTokenizer.from_pretrained("facebook/incoder-1B")

    print("creating lightseq model...")
    # XGLM shares the same model architecture as GPT
    ls_model = lsi.Gpt("lightseq_incoder_base.hdf5", max_batch_size=16)

    print("creating huggingface model...")
    hf_model = AutoModelForCausalLM.from_pretrained("facebook/incoder-1B")
    hf_model.to("cuda:0")

    # lightseq xglm perplexity supports batch infer with different lengths,
    # but sampling doesn't support
    sentences = ["def quick_sort(nums):"]

    print("====================START warmup====================")
    warmup(ls_tokenizer, hf_tokenizer, ls_model, hf_model, sentences)
    print("====================END warmup====================")

    while True:
        if args.user_input:
            sentences = [input("input the masked sentence:\n")]

        print("tokenizing the sentences...")

        ls_inputs = ls_tokenizer(sentences, return_tensors="pt")["input_ids"]
        hf_inputs = hf_tokenizer(sentences, return_tensors="pt")["input_ids"]

        ls_generate(ls_model, ls_tokenizer, ls_inputs)
        hf_generate(hf_model, hf_tokenizer, hf_inputs)

        if not args.user_input:
            break


if __name__ == "__main__":
    main()
