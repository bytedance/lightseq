import time
import argparse

import torch
import lightseq.inference as lsi
from transformers import BertTokenizer, BertModel


def ls_bert(model, inputs):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    last_hidden_state = model.infer(inputs)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return last_hidden_state, end_time - start_time


def hf_bert(model, inputs):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    last_hidden_state = model(inputs.to("cuda:0"))
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return last_hidden_state, end_time - start_time


def ls_generate(model, inputs_id):
    print("=========lightseq=========")
    print("lightseq generating...")
    ls_hidden_state, ls_time = ls_bert(model, inputs_id)
    print(f"lightseq time: {ls_time}s")
    print("lightseq results:")
    print(ls_hidden_state)


def hf_generate(model, inputs_id):
    print("=========huggingface=========")
    print("huggingface generating...")
    hf_output, hf_time = hf_bert(model, inputs_id)
    print(f"huggingface time: {hf_time}s")
    print("huggingface results:")
    print(hf_output.last_hidden_state.detach().cpu().numpy())


def warmup(tokenizer, ls_model, hf_model, sentences):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True)
    inputs_id = inputs["input_ids"]

    ls_generate(ls_model, inputs_id)
    hf_generate(hf_model, inputs_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_input", action="store_true")
    args = parser.parse_args()

    print("initializing bert tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    print("creating lightseq model...")
    ls_model = lsi.Bert("lightseq_bert_base_uncased.hdf5", 128)
    print("creating huggingface model...")
    hf_model = BertModel.from_pretrained("bert-base-uncased")
    hf_model.to("cuda:0")

    sentences = ["Hello, my dog is cute"]

    print("====================START warmup====================")
    warmup(tokenizer, ls_model, hf_model, sentences)
    print("====================END warmup====================")

    while True:
        if args.user_input:
            sentences = [input("input the masked sentence:\n")]

        print("tokenizing the sentences...")
        inputs = tokenizer(sentences, return_tensors="pt", padding=True)
        inputs_id = inputs["input_ids"]

        ls_generate(ls_model, inputs_id)
        hf_generate(hf_model, inputs_id)

        if not args.user_input:
            break


if __name__ == "__main__":
    main()
