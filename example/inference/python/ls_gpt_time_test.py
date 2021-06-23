import time
import torch
import lightseq.inference as lsi
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def ls_gpt(model, inputs):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    generated_ids = model.sample(inputs)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return generated_ids, end_time - start_time


def hf_gpt(model, inputs, tokenizer):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    generated_ids = model.generate(
        inputs, max_length=50, pad_token_id=tokenizer.eos_token_id
    )
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return generated_ids, end_time - start_time


def main():
    print("initializing gpt tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    print("creating lightseq model...")
    ls_model = lsi.Gpt(
        "lightseq_gpt2.pb",
        max_batch_size=128,
        max_step=50,  # max length
    )
    print("creating huggingface model...")
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")

    # GPU warm up
    sentences = [" ".join(["I"] * 10)] * 8
    inputs = tokenizer(sentences, return_tensors="pt", padding=True)
    inputs_id = inputs["input_ids"]
    _, _ = ls_gpt(ls_model, inputs_id)
    _, _ = hf_gpt(hf_model, inputs_id, tokenizer)

    bsz_list = [1, 2, 4, 8, 16, 32, 64, 128]
    seq_len_list = [1, 2, 4, 8, 16, 32]
    for bsz in bsz_list:
        total_ls = 0.0
        total_hf = 0.0
        for seq_len in seq_len_list:
            sentences = [" ".join(["I"] * seq_len)] * bsz
            inputs = tokenizer(sentences, return_tensors="pt", padding=True)
            inputs_id = inputs["input_ids"]
            _, ls_time = ls_gpt(ls_model, inputs_id)
            _, hf_time = hf_gpt(hf_model, inputs_id, tokenizer)
            total_ls += ls_time
            total_hf += hf_time
        print(f"{bsz}: {total_hf/total_ls-1}")


if __name__ == "__main__":
    main()
