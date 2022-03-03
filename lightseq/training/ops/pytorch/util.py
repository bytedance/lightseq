import math
from typing import Callable
import numpy as np
import torch
import torch.nn.functional as F


def softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)


def get_activation_fn(activation: str) -> Callable:
    """Returns the activation function corresponding to `activation`"""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


def copy_para(x):
    return torch.nn.Parameter(torch.empty_like(x).copy_(x))


def state_dict(module, destination=None, prefix="", keep_vars=False):
    destination = torch.nn.Module.state_dict(
        module, destination=destination, prefix=prefix, keep_vars=keep_vars
    )

    for key in destination.keys():
        if "para_16" in key:
            destination.pop(key)

    return destination


def check_config(config):
    if config.hidden_size % config.nhead != 0:
        raise Exception(f"hidden_size % nhead != 0")

    factor = 8 if config.fp16 else 4
    upbound = factor * 1024
    if config.hidden_size > upbound:
        # as required by ln backward kernel currently
        raise Exception(f"hidden_size > {upbound}")

    head_dim = config.hidden_size // config.nhead
    if head_dim % factor != 0:
        # as required by reshape kernel
        raise Exception(f"head_dim({head_dim}) % {factor} != 0")


def calc_offset(sizes):
    offsets = [0]
    tmp = 0
    for x in sizes:
        tmp += x
        offsets.append(tmp)
    return offsets


def get_pos_embedding(max_length, embedding_dim):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    emb = torch.arange(max_length, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(max_length, -1)
    if embedding_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(max_length, 1)], dim=1)
    return emb


def base_architecture(args):
    args.setdefault("hidden_size", 512)
    args.setdefault("intermediate_size", 2048)
    args.setdefault("nhead", 8)
    args.setdefault("attn_prob_dropout_ratio", 0.0)
    args.setdefault("activation_dropout_ratio", 0.0)
    args.setdefault("hidden_dropout_ratio", 0.1)
    args.setdefault("pre_layer_norm", True)
    args.setdefault("activation_fn", "relu")


def transformer_base(args):
    base_architecture(args)


def transformer_big(args):
    args.setdefault("hidden_size", 1024)
    args.setdefault("intermediate_size", 4096)
    args.setdefault("nhead", 16)
    args.setdefault("attn_prob_dropout_ratio", 0.1)
    args.setdefault("activation_dropout_ratio", 0.1)
    base_architecture(args)


def bert_base(args):
    args.setdefault("hidden_size", 768)
    args.setdefault("intermediate_size", 3072)
    args.setdefault("nhead", 12)
    args.setdefault("attn_prob_dropout_ratio", 0.1)
    args.setdefault("activation_dropout_ratio", 0.1)
    args.setdefault("pre_layer_norm", False)
    args.setdefault("activation_fn", "gelu")
    base_architecture(args)


def bert_big(args):
    args.setdefault("pre_layer_norm", False)
    args.setdefault("activation_fn", "gelu")
    transformer_big(args)


MODEL_ARCH = {
    "transformer-base": transformer_base,
    "transformer-big": transformer_big,
    "bert-base": bert_base,
    "bert-big": bert_big,
}
