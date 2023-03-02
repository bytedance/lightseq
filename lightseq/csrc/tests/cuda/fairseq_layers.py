import sys, os
import __init__

from typing import Dict, List, Optional, Callable
import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor

from csrc.pytorch import TransformerEncoderLayer, TransformerDecoderLayer


def generate_enc_layer():
    hidden_size = 1024
    intermediate_size = 1024 * 4
    heads = 16
    hidden_dropout_ratio = 0.0
    attn_dropout_ratio = 0.0
    activation_dropout_ratio = 0.0
    pre_layer_norm = True
    config = TransformerEncoderLayer.get_config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        nhead=heads,
        attn_prob_dropout_ratio=attn_dropout_ratio,
        activation_dropout_ratio=activation_dropout_ratio,
        hidden_dropout_ratio=hidden_dropout_ratio,
        pre_layer_norm=pre_layer_norm,
        activation_fn="relu",
        max_batch_tokens=None,
        max_seq_len=None,
        fp16=None,
        local_rank=None,
    )
    layer = TransformerEncoderLayer(config)
    layer.to(torch.device("cuda:0"), dtype=torch.half)
    return layer


def generate_dec_layer():
    hidden_size = 1024
    intermediate_size = 1024 * 4
    heads = 16
    hidden_dropout_ratio = 0.0
    attn_dropout_ratio = 0.0
    activation_dropout_ratio = 0.0
    pre_layer_norm = True
    config = TransformerDecoderLayer.get_config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        nhead=heads,
        attn_prob_dropout_ratio=attn_dropout_ratio,
        activation_dropout_ratio=activation_dropout_ratio,
        hidden_dropout_ratio=hidden_dropout_ratio,
        pre_layer_norm=pre_layer_norm,
        activation_fn="relu",
        max_batch_tokens=None,
        max_seq_len=None,
        fp16=None,
        local_rank=None,
        nlayer=None,
    )
    layer = TransformerDecoderLayer(config)

    layer.to(torch.device("cuda:0"), dtype=torch.half)
    return layer


def generate_bert_enc_layer():
    hidden_size = 1024
    intermediate_size = 1024 * 4
    heads = 16
    hidden_dropout_ratio = 0.0
    attn_dropout_ratio = 0.0
    activation_dropout_ratio = 0.0
    config = TransformerEncoderLayer.get_config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        nhead=heads,
        attn_prob_dropout_ratio=attn_dropout_ratio,
        activation_dropout_ratio=activation_dropout_ratio,
        hidden_dropout_ratio=hidden_dropout_ratio,
        pre_layer_norm=False,
        activation_fn="gelu",
        max_batch_tokens=None,
        max_seq_len=None,
        fp16=None,
        local_rank=None,
    )
    layer = TransformerEncoderLayer(config)
    layer.to(torch.device("cuda:0"), dtype=torch.half)
    return layer


# def generate_emb_layer(ls_emb_config):
#     layer = TransformerEmbeddingLayer(ls_emb_config)
#     dtype = torch.float16 if ls_emb_config.fp16 else torch.float32
#     layer.to(torch.device("cuda:0"), dtype=dtype)

#     return layer


if __name__ == "__main__":
    generate_enc_layer()
    # generate_dec_layer()
