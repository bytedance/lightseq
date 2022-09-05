import time
import argparse

import torch
import lightseq.inference as lsi
from transformers import FSMTForConditionalGeneration, FSMTTokenizer


def ls_bart(model, inputs):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    generated_ids = model.infer(inputs)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return generated_ids, end_time - start_time


def hf_bart(model, inputs):
    output_seq_len = inputs.size()[1]
    print("hf_bart input size:", inputs.size())
    print("output seq len", output_seq_len)
    inputs = inputs.to("cuda:0")
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(20):
        generated_ids = model.generate(
            inputs,
            max_length=output_seq_len,
            min_length=output_seq_len,
            use_cache=True,
            early_stopping=False,
            num_beams=4,
        )
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    print("output size:", generated_ids.size())
    return generated_ids, (end_time - start_time) / 20


def ls_generate(model, tokenizer, inputs_id):
    print("=========lightseq=========")
    print("lightseq generating...")
    ls_res_ids, ls_time = ls_bart(model, inputs_id)
    ls_res_ids = [ids[0] for ids in ls_res_ids[0]]
    ls_res = tokenizer.batch_decode(ls_res_ids, skip_special_tokens=True)
    print(f"lightseq time: {ls_time}s")
    print("lightseq results:")
    for sent in ls_res:
        print(sent)


def hf_generate(model, tokenizer, inputs_id):
    print("=========huggingface=========")
    print("huggingface generating...")
    print(inputs_id.size())
    hf_res_ids, hf_time = hf_bart(model, inputs_id)
    hf_res = tokenizer.batch_decode(hf_res_ids, skip_special_tokens=True)
    print(f"huggingface time: {hf_time}s")
    print("huggingface results:")
    for sent in hf_res:
        print(sent)


def warmup(tokenizer, ls_model, hf_model, sentences):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True)
    inputs_id = inputs["input_ids"]

    # ls_generate(ls_model, tokenizer, inputs_id)
    hf_generate(hf_model, tokenizer, inputs_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_input", action="store_true")
    args = parser.parse_args()

    print("initializing bart tokenizer...")
    # change to "facebook/bart-large" for large model
    tokenizer = FSMTTokenizer.from_pretrained("facebook/wmt19-en-de")

    print("creating lightseq model...")
    # change to "lightseq_bart_large.hdf5" for large model
    # ls_model = lsi.Transformer("lightseq_bart_base.hdf5", 128)
    ls_model = None
    print("creating huggingface model...")
    # change to "facebook/bart-large" for large model
    hf_model = FSMTForConditionalGeneration.from_pretrained("facebook/wmt19-en-de")
    hf_model.to("cuda:0")
    hf_model.eval()
    hf_model.half()

    sentences = [
        "I love that girl, but she does not love me.",
    ]

    print("====================START warmup====================")
    warmup(tokenizer, ls_model, hf_model, sentences)
    print("====================END warmup====================")

    while True:
        # if args.user_input:
        #     sentences = [input("input the masked sentence:\n")]

        print("tokenizing the sentences...")
        inputs = tokenizer(sentences, return_tensors="pt", padding=True)
        inputs_id = inputs["input_ids"]
        input_len = inputs_id.size()[1]
        batch_size = 32
        input_seq_len = output_seq_len = 64
        repeat_factor = (input_seq_len // input_len + 1) * input_len

        inputs_id = inputs_id.repeat(batch_size, repeat_factor)
        inputs_id = inputs_id[:, :input_seq_len]
        print("input size:", inputs_id.size())
        hf_generate(hf_model, tokenizer, inputs_id)

        if not args.user_input:
            break


if __name__ == "__main__":
    main()
