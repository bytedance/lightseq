import time
import argparse

import torch
import lightseq.inference as lsi
from transformers import BertTokenizer, BertForSequenceClassification


def ls_bert(model, inputs, attn_mask):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    ls_output = model.infer(inputs, attn_mask)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return ls_output, end_time - start_time


def hf_bert(model, inputs, attn_mask):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    hf_output = model(inputs.to("cuda:0"), attention_mask=attn_mask.to("cuda:0"))
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return hf_output, end_time - start_time


def ls_generate(model, inputs_id, attn_mask):
    print("=========lightseq=========")
    print("lightseq generating...")
    ls_output, ls_time = ls_bert(model, inputs_id, attn_mask)
    print(f"lightseq time: {ls_time}s")
    print("lightseq results (class predictions):")
    print(ls_output.argmax(axis=1).detach().cpu().numpy())


def hf_generate(model, inputs_id, attn_mask):
    print("=========huggingface=========")
    print("huggingface generating...")
    hf_output, hf_time = hf_bert(model, inputs_id, attn_mask)
    print(f"huggingface time: {hf_time}s")
    print("huggingface results (class predictions):")
    print(hf_output.logits.argmax(axis=1).detach().cpu().numpy())


def warmup(tokenizer, ls_model, hf_model, sentences):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True)
    inputs_id = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]

    ls_generate(ls_model, inputs_id, attn_mask)
    hf_generate(hf_model, inputs_id, attn_mask)


class LightseqBertClassification:
    def __init__(self, ls_weight_path, hf_model):
        self.ls_bert = lsi.Bert(ls_weight_path, 128)
        self.pooler = hf_model.bert.pooler
        self.classifier = hf_model.classifier

    def infer(self, inputs, attn_mask):
        last_hidden_states = self.ls_bert.infer(inputs, attn_mask)
        last_hidden_states = torch.Tensor(last_hidden_states).float()
        pooled_output = self.pooler(last_hidden_states.to("cuda:0"))
        logits = self.classifier(pooled_output)
        return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_input", action="store_true")
    args = parser.parse_args()

    print("initializing bert tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    print("creating huggingface model...")
    hf_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    hf_model.to("cuda:0")

    print("creating lightseq model...")
    ls_model = LightseqBertClassification("lightseq_bert_base_uncased.hdf5", hf_model)

    sentences = [
        "Hello, my dog is cute",
        "Hey, how are you",
        "This is a test",
        "Testing the model again",
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
        attn_mask = inputs["attention_mask"]

        ls_generate(ls_model, inputs_id, attn_mask)
        hf_generate(hf_model, inputs_id, attn_mask)

        if not args.user_input:
            break


if __name__ == "__main__":
    main()
