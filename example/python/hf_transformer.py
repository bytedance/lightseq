import time
import torch
import lightseq
from transformers import BartTokenizer, BartForConditionalGeneration


def lightseq_hf(inputs):
    model = lightseq.Transformer("lightseq_bart_base.pb", 128)
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    generated_ids = model.infer(inputs)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return generated_ids, end_time-start_time

def raw_hf(inputs):
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    generated_ids = model.generate(inputs)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return generated_ids, end_time-start_time

def main():
    print("initializing bart tokenizer...")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    sentences = [
        "I love that girl, but <mask> does not <mask> me.",
        "She told me that <mask> loved another <mask>.",
        "She is so <mask> that I can not help glance at <mask>.",
        "How can <mask> say that you are <mask> than me?",
        "I am going all the <mask> north, leaving the <mask> with you.",
        "I am your <mask> and you are my son.",
        "Would you like to be my <mask>? I will give all my <mask> to you."
    ]
    print("tokenizing the sentences...")
    inputs = tokenizer(sentences, return_tensors="pt", padding=True)
    inputs_id = inputs["input_ids"]

    print("=========================lightseq=========================")
    print("lightseq generating...")
    ls_res_ids, ls_time = lightseq_hf(inputs_id)
    ls_res_ids = [ids[0] for ids in ls_res_ids[0]]
    ls_res = tokenizer.batch_decode(ls_res_ids, skip_special_tokens=True)
    print(f"lightseq time: {ls_time}s\nlightseq results:")
    for sent in ls_res:
        print(sent)

    print("=========================huggingface=========================")
    print("huggingface generating...")
    hf_res_ids, hf_time = raw_hf(inputs_id)
    hf_res = tokenizer.batch_decode(hf_res_ids, skip_special_tokens=True)
    print(f"huggingface time: {hf_time}s\nhuggingface results:")
    for sent in hf_res:
        print(sent)

if __name__ == "__main__":
    main()
