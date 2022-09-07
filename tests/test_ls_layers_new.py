import random
import math
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn


from tests.util import (
    TestDecorator,
    get_fairseq_enc_params,
    get_fairseq_dec_params,
    max_batch_tokens,
    max_seq_len,
    split_custom_layer_grad,
    copy_grad_from_paras,
)

from tests import fairseq_layers
from lightseq.training.ops.pytorch.transformer_encoder_layer_new import (
    LSTransformerEncoderLayerNew,
)

from lightseq.training.ops.pytorch.transformer_encoder_layer import (
    LSTransformerEncoderLayer,
)

kt = TestDecorator()


def generate_enc_layer_new(initial_weights=None, initial_biases=None):
    config = LSTransformerEncoderLayerNew.get_config(
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
    layer = LSTransformerEncoderLayerNew(config, initial_weights, initial_biases)
    layer.to(torch.device("cuda:0"), dtype=torch.half)
    return layer


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

    custom_enc_layer_base = generate_enc_layer(initial_enc_weights, initial_enc_biases)
    custom_enc_layer_base.train()

    custom_enc_layer_new = generate_enc_layer_new(
        initial_enc_weights, initial_enc_biases
    )
    custom_enc_layer_new.train()

    return custom_enc_layer_base, custom_enc_layer_new


ENC_LAYER_NUM = 2
base_enc_layers = []
custom_enc_layers = []

for _ in range(ENC_LAYER_NUM):
    base_enc, custom_enc = gen_enc_layer_pair()
    base_enc_layers.append(base_enc)
    custom_enc_layers.append(custom_enc)


@kt.case(dtypes=[torch.half], rtol=1e-3, atol=1e-2, ntest=5, nrepeat=5)
def test_encoder_layer_forward():
    batch_size, seq_len = kt.bs_sl()
    print(f"(batch_size, seq_len): ({batch_size}, {seq_len})")

    hidden_states = kt.rand((batch_size, seq_len, 1024))
    self_attn_padding_mask = kt.attn_mask(batch_size, seq_len, dtype=torch.bool)

    def custom():
        res = hidden_states.clone()
        for i in range(ENC_LAYER_NUM):
            res = custom_enc_layers[i](res, self_attn_padding_mask)
        return [
            res.contiguous().detach(),
        ]

    def baseline():
        res = hidden_states.clone()
        for i in range(ENC_LAYER_NUM):
            res = base_enc_layers[i](res, self_attn_padding_mask)
        return [
            res.contiguous().detach(),
        ]

    return custom, baseline


@kt.case(dtypes=[torch.half], rtol=1e-3, atol=1e-2, ntest=5, nrepeat=5)
def test_encoder_layer_backward():
    batch_size, seq_len = kt.bs_sl()
    print(f"(batch_size, seq_len): ({batch_size}, {seq_len})")

    hidden_size = 1024

    hidden_states = kt.rand((batch_size, seq_len, hidden_size)).requires_grad_()
    self_attn_padding_mask = kt.attn_mask(batch_size, seq_len, dtype=torch.bool)
    loss_data = torch.randn(1, dtype=hidden_states.dtype).sum()

    def custom():
        for i in range(ENC_LAYER_NUM):
            custom_enc_layers[i].zero_grad()
        res = hidden_states.clone()
        for i in range(ENC_LAYER_NUM):
            res = custom_enc_layers[i](res, self_attn_padding_mask)
        custom_loss = (res / 1000).sum()
        custom_loss.data.copy_(loss_data)
        custom_loss.backward()
        grad_list = []
        for i in range(ENC_LAYER_NUM - 1, -1, -1):
            """
            attn_qkvw, attn_qkvb, attn_ow, attn_ob, attn_nw, attn_nb,
            inter_w, inter_b, output_w, output_b, ffn_nw, ffn_nb
            """
            grads = split_custom_layer_grad(custom_enc_layers[i])
            grad_list.extend(grads[:11])
            pass
        return grad_list

    def baseline():
        for i in range(ENC_LAYER_NUM):
            base_enc_layers[i].zero_grad()
        res = hidden_states.clone()
        for i in range(ENC_LAYER_NUM):
            res = base_enc_layers[i](res, self_attn_padding_mask)
        custom_loss = (res / 1000).sum()
        custom_loss.data.copy_(loss_data)
        custom_loss.backward()
        grad_list = []
        for i in range(ENC_LAYER_NUM - 1, -1, -1):
            """
            attn_qkvw, attn_qkvb, attn_ow, attn_ob, attn_nw, attn_nb,
            inter_w, inter_b, output_w, output_b, ffn_nw, ffn_nb
            """
            grads = split_custom_layer_grad(base_enc_layers[i])
            grad_list.extend(grads[:11])
            pass
        return grad_list

    return custom, baseline


if __name__ == "__main__":

    kt.init(device="cuda:0", nhead=16)
    kt.run(["test_encoder_layer_forward", "test_encoder_layer_backward"])
