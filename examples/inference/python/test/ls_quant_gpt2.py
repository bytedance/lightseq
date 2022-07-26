import time

import torch
from torch import nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import lightseq.inference as lsi
from lightseq.training.ops.pytorch.quantization import (
    qat_mode,
    QuantLinear,
    TensorQuantizer,
    weight_quant_config,
    emb_quant_config,
)
from lightseq.training.ops.pytorch.torch_transformer_layers import (
    TransformerDecoderLayer,
)
from export.util import parse_args


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
            inputs, max_length=50, pad_token_id=tokenizer.eos_token_id
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
    ls_inputs = ls_tokenizer(sentences, return_tensors="pt", padding=True)["input_ids"]
    hf_inputs = hf_tokenizer(sentences, return_tensors="pt", padding=True)["input_ids"]

    if generation_method == "topk" or generation_method == "topp":
        ls_generate(ls_model, ls_tokenizer, ls_inputs)
        # hf_generate(hf_model, hf_tokenizer, hf_inputs)
    elif generation_method == "ppl":
        ls_ppl(ls_model, ls_tokenizer, ls_inputs)
        hf_ppl(hf_model, hf_tokenizer, hf_inputs)


class GptEmbedding(nn.Embedding):
    def __init__(self, *args, **kwargs):
        super(GptEmbedding, self).__init__(*args, **kwargs)
        self.emb_quant = TensorQuantizer(emb_quant_config)

    def forward(self, input_ids):
        x = super(GptEmbedding, self).forward(input_ids)
        x = self.emb_quant(x)
        return x


def gen_gpt_enc_config(config):
    gpt_enc_config = TransformerDecoderLayer.get_config(
        max_batch_tokens=8192,
        max_seq_len=config.max_position_embeddings,
        hidden_size=config.hidden_size,
        intermediate_size=4 * config.hidden_size,
        nhead=config.num_attention_heads,
        attn_prob_dropout_ratio=config.attn_pdrop,
        activation_dropout_ratio=config.resid_pdrop,
        hidden_dropout_ratio=config.resid_pdrop,
        pre_layer_norm=True,
        fp16=True,
        local_rank=0,
        nlayer=config.num_hidden_layers,
        activation_fn="gelu",
        has_cross_attn=False,
    )
    return gpt_enc_config


class LSHFGptEncoderLayer(TransformerDecoderLayer):
    def __init__(self, *args, **kwargs):
        super(LSHFGptEncoderLayer, self).__init__(*args, **kwargs)

    def forward(self, hidden_states, attention_mask=None, *args, **kwargs):
        if attention_mask is not None:
            ls_attention_mask = attention_mask.squeeze()
        else:
            ls_attention_mask = torch.zeros(hidden_states.size()[:2])
        output = super().forward(hidden_states, ls_attention_mask)
        return output


def inject_ls_layer(model, config):
    model.transformer.wte = GptEmbedding(config.vocab_size, config.hidden_size)
    model.transformer.wte.apply(qat_mode)

    for i in range(config.num_hidden_layers):
        gpt_enc_config = gen_gpt_enc_config(config)
        model.transformer.h[i] = LSHFGptEncoderLayer(gpt_enc_config).cuda()
        model.transformer.h[i].apply(qat_mode)

    q_lm_head = QuantLinear(config.n_embd, config.vocab_size, bias=False)
    q_lm_head.weight = model.transformer.wte.weight
    q_lm_head.weight_quant = model.transformer.wte.emb_quant
    model.lm_head = q_lm_head


def main():
    args = parse_args()
    if args.generation_method not in ["topk", "topp", "ppl"]:
        args.generation_method = "topk"
    model_name = ".".join(args.model.split(".")[:-1])
    ckpt_path = f"{model_name}.bin"

    print("initializing gpt2 config...")
    config = GPT2Config.from_pretrained("gpt2")

    print("initializing gpt2 tokenizer...")
    ls_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # lightseq use len(tokenizer) as pad_token in default
    ls_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    print(f"lightseq tokenizer pad token id: {ls_tokenizer.pad_token_id}")

    hf_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # use EOS as PAD for huggingface to avoid warning according to https://huggingface.co/blog/how-to-generate while avoid reshaping the model embedding
    hf_tokenizer.pad_token = hf_tokenizer.eos_token
    print(f"huggingface tokenizer pad token id: {hf_tokenizer.pad_token_id}")

    print("creating huggingface model...")
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
    inject_ls_layer(hf_model, config)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    hf_model.load_state_dict(state_dict, strict=False)
    hf_model.to("cuda:0")
    hf_model.eval()

    print("creating lightseq model...")
    ls_model = lsi.QuantGpt(args.model, max_batch_size=16)

    # lightseq gpt perplexity supports batch infer with different lengths,
    # but sampling doesn't support
    sentences = [
        "I love you, but you say that",
        "I love you, but you say that",
        "I love you, but you say that",
        "I love you, but you say that",
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

    print("tokenizing the sentences...")
    ls_inputs = ls_tokenizer(sentences, return_tensors="pt", padding=True)["input_ids"]
    hf_inputs = hf_tokenizer(sentences, return_tensors="pt", padding=True)["input_ids"]

    if args.generation_method == "topk" or args.generation_method == "topp":
        ls_generate(ls_model, ls_tokenizer, ls_inputs)
        # hf_generate(hf_model, hf_tokenizer, hf_inputs)
    elif args.generation_method == "ppl":
        ls_ppl(ls_model, ls_tokenizer, ls_inputs)
        hf_ppl(hf_model, hf_tokenizer, hf_inputs)


if __name__ == "__main__":
    main()
