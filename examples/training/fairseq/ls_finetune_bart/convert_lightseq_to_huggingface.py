"""Convert BART checkpoint."""


import argparse
from pathlib import Path

import fairseq
import torch
from packaging import version
from torch import nn

from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartForSequenceClassification,
    BartModel,
    BartTokenizer,
)
from lightseq.training.cli.fs_modules import LSBARTModel
from fairseq.models.bart import BARTModel

if version.parse(fairseq.__version__) < version.parse("0.9.0"):
    raise Exception("requires fairseq >= 0.9.0")

SAMPLE_TEXT = " Hello world! cécé herlolip"


def rename_key(dct, old, new):
    assert old in dct, f"{old} not in dct!"
    val = dct.pop(old)
    dct[new] = val


def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


def split_checkpoint_path(checkpoint_path):
    path = Path(checkpoint_path)
    return path.parent, path.name


def _upgrade_huggingface_bart_base_state_dict(state_dict, arch, hf_config):
    embed_weight = state_dict["encoder.embed_tokens.weight"]
    offset = hf_config.vocab_size - embed_weight.shape[0]
    if offset > 0:
        h = torch.zeros(offset, embed_weight.shape[1]).to(embed_weight)

    def truncate_emb(key):
        if key in state_dict:
            if offset > 0:
                state_dict[key] = torch.cat((state_dict[key], h), dim=0)
            else:
                state_dict[key] = state_dict[key][:offset, :]

    if offset != 0:
        truncate_emb("encoder.embed_tokens.weight")
        truncate_emb("decoder.embed_tokens.weight")
        truncate_emb("encoder.output_projection.weight")
        truncate_emb("decoder.output_projection.weight")

    state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]


def test_lightseq_bart(checkpoint_path):
    dirname, modelname = split_checkpoint_path(checkpoint_path)

    args = {
        "arch": "ls_bart_base",
    }
    fs_bart = (
        BARTModel.from_pretrained(dirname, checkpoint_file=modelname, fp16=False)
        .to("cuda:0")
        .eval()
    )
    ls_bart = (
        LSBARTModel.from_pretrained(
            dirname, checkpoint_file=modelname, fp16=False, **args
        )
        .to("cuda:0")
        .eval()
    )

    tokens = fs_bart.encode(SAMPLE_TEXT).unsqueeze(0).to("cuda:0")

    fs_output = fs_bart.extract_features(tokens)
    ls_output = ls_bart.extract_features(tokens)

    assert torch.allclose(
        fs_output.flatten(), ls_output.flatten(), rtol=1e-1, atol=1e-1, equal_nan=False
    )


def _upgrade_pytorch_state_dict(state_dict, args, model):
    # remove lightseq keys
    remove_suffix = [
        "_amax",
        "para",
        "embeddings",
        "decoder.output_projection.weight",
    ]
    for k in list(state_dict.keys()):
        for suf in remove_suffix:
            if k.endswith(suf):
                del state_dict[k]

    # update state dict of embeddings
    embed_map_dict = {
        "encoder.embed_tokens.emb_lookup.weight": "encoder.embed_tokens.weight",
        "encoder.embed_tokens.embed_positions.weight": "encoder.embed_positions.weight",
        "encoder.embed_tokens.layernorm_embedding.weight": "encoder.layernorm_embedding.weight",
        "encoder.embed_tokens.layernorm_embedding.bias": "encoder.layernorm_embedding.bias",
        "decoder.embed_tokens.emb_lookup.weight": "decoder.embed_tokens.weight",
        "decoder.embed_tokens.embed_positions.weight": "decoder.embed_positions.weight",
        "decoder.embed_tokens.layernorm_embedding.weight": "decoder.layernorm_embedding.weight",
        "decoder.embed_tokens.layernorm_embedding.bias": "decoder.layernorm_embedding.bias",
    }
    for k, v in embed_map_dict.items():
        rename_key(state_dict, k, v)

    # update state dict of encoder
    for lid in range(args.encoder_layers):
        prefix = f"encoder.layers.{lid}."
        w, b = model.encoder.layers[lid].params_dict()
        for k, v in w.items():
            state_dict[prefix + k + ".weight"] = v
        for k, v in b.items():
            state_dict[prefix + k + ".bias"] = v

    # update state dict of decoder
    skip_k = ["encoder_attn.k_proj", "encoder_attn.v_proj"]
    enc_attn_kv = {}
    for lid in range(args.decoder_layers):
        prefix = f"decoder.layers.{lid}."
        w, b = model.decoder.layers[lid].params_dict()
        for k, v in w.items():
            if k not in skip_k:
                state_dict[prefix + k + ".weight"] = v
        for k, v in b.items():
            if k not in skip_k:
                state_dict[prefix + k + ".bias"] = v
        if lid == 0:
            enc_attn_kv["encoder_attn.k_proj.weight"] = w["encoder_attn.k_proj"]
            enc_attn_kv["encoder_attn.k_proj.bias"] = b["encoder_attn.k_proj"]
            enc_attn_kv["encoder_attn.v_proj.weight"] = w["encoder_attn.v_proj"]
            enc_attn_kv["encoder_attn.v_proj.bias"] = b["encoder_attn.v_proj"]
        for k, v in enc_attn_kv.items():
            state_dict[prefix + k] = v[lid]


def load_huggingface_model(state_dict, config, hf_checkpoint_name="facebook/bart-base"):
    model = BartForConditionalGeneration(config).eval()
    model.model.load_state_dict(state_dict)
    if hasattr(model, "lm_head"):
        model.lm_head = make_linear_from_emb(model.model.shared)
    return model


@torch.no_grad()
def convert_ls2hf(checkpoint_path, hf_checkpoint_name, pytorch_dump_folder_path):
    device = "cuda:0"
    dirname, modelname = split_checkpoint_path(checkpoint_path)

    hf_config = BartConfig.from_pretrained(hf_checkpoint_name)
    ls_model = (
        LSBARTModel.from_pretrained(dirname, checkpoint_file=modelname, fp16=False)
        .to(device)
        .eval()
    )
    args, model = ls_model.args, ls_model.model
    state_dict = ls_model.model.state_dict()

    if args.task != "translation":
        ValueError("task name should be translation")
    if args.arch.startswith("ls"):
        ValueError("Only lightseq model is supported")

    
    # upgrade state dict
    _upgrade_pytorch_state_dict(state_dict, args, model)
    _upgrade_huggingface_bart_base_state_dict(state_dict, args.arch, hf_config)

    # load huggface model
    hf_model = load_huggingface_model(state_dict, hf_config, hf_checkpoint_name).to(device)

    # forward
    tokens = ls_model.encode(SAMPLE_TEXT).unsqueeze(0).to(device)
    tokens2 = (
        BartTokenizer.from_pretrained(hf_checkpoint_name)
        .encode(SAMPLE_TEXT, return_tensors="pt")
        .unsqueeze(0)
        .to(device)
    )

    ls_outputs = ls_model.extract_features(tokens)
    hf_outputs = hf_model.model(tokens)[0]

    # Check results
    assert torch.eq(tokens, tokens2).all()
    assert ls_outputs.shape == hf_outputs.shape
    assert torch.allclose(
        ls_outputs.flatten(),
        hf_outputs.flatten(),
        rtol=1e-1,
        atol=1e-1,
        equal_nan=False,
    )
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    hf_model.save_pretrained(pytorch_dump_folder_path)
    print(f"saved model and config in {pytorch_dump_folder_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--fairseq_path",
        type=str,
        help="bart.large, bart.large.cnn or a path to a model.pt on local filesystem.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model.",
    )
    parser.add_argument(
        "--hf_config",
        default=None,
        type=str,
        help="Which huggingface architecture to use: bart-large-xsum",
    )
    args = parser.parse_args()
    convert_ls2hf(args.fairseq_path, args.hf_config, args.pytorch_dump_folder_path)
