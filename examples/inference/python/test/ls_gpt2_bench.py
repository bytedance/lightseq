import time
import argparse

import torch
import lightseq.inference as lsi
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def ls_gpt2(model, inputs, generation_method="topk"):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    results = None
    if generation_method == "topk" or generation_method == "topp":
        results = model.sample(inputs)
    elif generation_method == "ppl":
        results = model.ppl(inputs)[0]
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return results, end_time - start_time


def compute_hf_ppl(model, inputs):
    max_length = 512
    stride = 512
    end_loc = 0

    nlls = []
    for i in range(0, inputs.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, inputs.size(1))
        trg_len = end_loc - i
        input_ids = inputs[:, begin_loc:end_loc].to("cuda:0")
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)

    ppl = torch.stack(nlls).sum() / end_loc
    return ppl.cpu().numpy()


def hf_gpt2(model, inputs, tokenizer, generation_method="topk"):
    inputs = inputs.to("cuda:0")
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    results = None
    if generation_method == "topk" or generation_method == "topp":
        results = model.generate(
            inputs,
            min_length=150,
            max_length=150,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            top_k=1,
            early_stopping=False,
        )
    elif generation_method == "ppl":
        results = compute_hf_ppl(model, inputs)

    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return results, end_time - start_time


def ls_generate(model, tokenizer, inputs):
    print("=========lightseq=========")
    print("lightseq generating...")
    ls_res_ids, ls_time = ls_gpt2(model, inputs)
    ls_res = tokenizer.batch_decode(ls_res_ids, skip_special_tokens=True)
    print(f"lightseq time: {ls_time}s")
    print("lightseq results:")
    for sent in ls_res:
        print(sent)


def hf_generate(model, tokenizer, inputs):
    print("=========huggingface=========")
    print("huggingface generating...")
    hf_res_ids, hf_time = hf_gpt2(model, inputs, tokenizer)
    print("hf output size:", hf_res_ids.size())
    hf_res = tokenizer.batch_decode(hf_res_ids, skip_special_tokens=True)
    print(f"huggingface time: {hf_time}s")
    print("huggingface results:")
    for sent in hf_res:
        print(sent)


def ls_ppl(model, tokenizer, inputs):
    print("=========lightseq=========")
    print("lightseq calculating ppl...")
    ls_ppl, ls_time = ls_gpt2(model, inputs, "ppl")
    print(f"lightseq time: {ls_time}s")
    print("lightseq results:")
    print(ls_ppl)


def hf_ppl(model, tokenizer, inputs):
    print("=========huggingface=========")
    print("huggingface calculating ppl...")
    hf_ppl, hf_time = hf_gpt2(model, inputs, tokenizer, "ppl")
    print(f"huggingface time: {hf_time}s")
    print("huggingface results:")
    print(hf_ppl)


def warmup(
    ls_tokenizer, hf_tokenizer, ls_model, hf_model, sentences, generation_method
):
    # ls_inputs = ls_tokenizer(sentences, return_tensors="pt", padding=True)["input_ids"]
    hf_inputs = hf_tokenizer(sentences, return_tensors="pt", padding=True)["input_ids"]

    if generation_method == "topk" or generation_method == "topp":
        # ls_generate(ls_model, ls_tokenizer, ls_inputs)
        hf_generate(hf_model, hf_tokenizer, hf_inputs)
    elif generation_method == "ppl":
        # ls_ppl(ls_model, ls_tokenizer, ls_inputs)
        hf_ppl(hf_model, hf_tokenizer, hf_inputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_input", action="store_true")
    parser.add_argument(
        "--generation_method",
        "-g",
        type=str,
        default="topk",
        choices=["topk", "topp", "ppl"],
        help="generation method",
    )
    args = parser.parse_args()

    print("initializing gpt tokenizer...")

    ls_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # lightseq use len(tokenizer) as pad_token in default
    ls_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    print(f"lightseq tokenizer pad token id: {ls_tokenizer.pad_token_id}")

    hf_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # use EOS as PAD for huggingface to avoid warning according to https://huggingface.co/blog/how-to-generate while avoid reshaping the model embedding
    hf_tokenizer.pad_token = hf_tokenizer.eos_token
    print(f"huggingface tokenizer pad token id: {hf_tokenizer.pad_token_id}")

    print("creating lightseq model...")
    # ls_model = lsi.Gpt("lightseq_gpt2_base.hdf5", max_batch_size=16)
    ls_model = None

    print("creating huggingface model...")
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_model.to("cuda:0")
    hf_model.eval()
    hf_model.half()

    # lightseq gpt perplexity supports batch infer with different lengths,
    # but sampling doesn't support
    sentences = [
        "I love you, but you say that",
        # "I love you, but you say that",
        # "I love you, but you say that",
        # "I love you, but you say that",
    ]

    print("====================START warmup====================")
    warmup(
        ls_tokenizer,
        hf_tokenizer,
        ls_model,
        hf_model,
        sentences,
        args.generation_method,
    )
    print("====================END warmup====================")

    while True:
        if args.user_input:
            sentences = [input("input the masked sentence:\n")]

        print("tokenizing the sentences...")

        ls_inputs = ls_tokenizer(sentences, return_tensors="pt", padding=True)[
            "input_ids"
        ]
        hf_inputs = hf_tokenizer(sentences, return_tensors="pt", padding=True)[
            "input_ids"
        ]

        input_len = hf_inputs.size()[1]
        batch_size = 32
        input_seq_len = 22  # 118 86 22
        repeat_factor = (input_seq_len // input_len + 1) * input_len

        hf_inputs = hf_inputs.repeat(batch_size, repeat_factor)
        hf_inputs = hf_inputs[:, :input_seq_len]
        print("hf input size:", hf_inputs.size())
        if args.generation_method == "topk" or args.generation_method == "topp":
            # ls_generate(ls_model, ls_tokenizer, ls_inputs)
            hf_generate(hf_model, hf_tokenizer, hf_inputs)
        elif args.generation_method == "ppl":
            # ls_ppl(ls_model, ls_tokenizer, ls_inputs)
            hf_ppl(hf_model, hf_tokenizer, hf_inputs)

        if not args.user_input:
            break


if __name__ == "__main__":
    main()
