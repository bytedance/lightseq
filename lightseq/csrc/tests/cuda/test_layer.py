from lib2to3.pgen2.token import NUMBER
import sys
from __init__ import lightseq_dir

sys.path.insert(0, lightseq_dir)

import random
import math
from copy import deepcopy
from dataclasses import dataclass
import fairseq_layers
from csrc.tests.util import (
    TestDecorator,
    get_fairseq_enc_params,
    get_fairseq_dec_params,
    max_batch_tokens,
    max_seq_len,
    split_custom_layer_grad,
    copy_grad_from_paras,
    copy_cmax_grad_from_paras,
)

import torch
import torch.nn as nn

from csrc.pytorch.transformer_encoder_layer import LSTransformerEncoderLayer

kt = TestDecorator()

def generate_enc_layer(initial_weights=None, initial_biases=None):
    config = LSTransformerEncoderLayer.get_config(
        max_batch_tokens=max_batch_tokens,
        max_seq_len=max_seq_len,
        hidden_size=1024,
        intermediate_size=4096,
        nhead=16,
        attn_prob_dropout_ratio=0.0,
        activation_dropout_ratio=0.0,
        hidden_dropout_ratio=0.0,
        pre_layer_norm=True,
        fp16=True,
        local_rank=0,
        activation_fn="relu",
    )
    layer = LSTransformerEncoderLayer(config, initial_weights, initial_biases)
    layer.to(torch.device("cuda:0"), dtype=torch.half)
    return layer


def gen_enc_layer_pair():
    fairseq_enc_layer = fairseq_layers.generate_enc_layer()
    fairseq_enc_layer.train()
    initial_enc_weights, initial_enc_biases = get_fairseq_enc_params(fairseq_enc_layer)
    custom_enc_layer = generate_enc_layer(initial_enc_weights, initial_enc_biases)
    custom_enc_layer.train()
    return fairseq_enc_layer, custom_enc_layer


NUM_LAYERS = 4

custom_enc_layers = []
base_enc_layers = []

for _ in range(NUM_LAYERS):
    base_enc_layer, custom_enc_layer = gen_enc_layer_pair()
    custom_enc_layers.append(custom_enc_layer)
    base_enc_layers.append(base_enc_layer)


@kt.case(dtypes=[torch.half], rtol=1e-3, atol=1e-2, ntest=5, nrepeat=5)
def test_encoder_layer_forward():
    batch_size, seq_len = kt.bs_sl()
    print(f"(batch_size, seq_len): ({batch_size}, {seq_len})")

    hidden_states = kt.rand((batch_size, seq_len, 1024))
    self_attn_padding_mask = kt.attn_mask(batch_size, seq_len, dtype=torch.bool)

    def custom():
        res = hidden_states.clone()
        for i in range(NUM_LAYERS):
            res = custom_enc_layers[i](res, self_attn_padding_mask)
        return [
            res.contiguous().detach(),
        ]

    def baseline():
        res = hidden_states.clone()
        for i in range(NUM_LAYERS):
            res = base_enc_layers[i](res, self_attn_padding_mask)
        return [
            res.contiguous().detach(),
        ]

    return custom, baseline


if __name__ == "__main__":
    kt.init(device="cuda:0", nhead=16)
    kt.run(
        [
            "test_encoder_layer_forward",
        ]
    )
