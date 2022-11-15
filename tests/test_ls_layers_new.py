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
from lightseq.training.ops.pytorch import layer_cuda_module

from lightseq.training.cli.fs_modules.ls_fs_transformer_decoder_layer import (
    LSFSTransformerDecoderLayerNew,
    LSFSTransformerDecoderLayer,
)

kt = TestDecorator()


###################### generate encoder layers ######################


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


NUM_LAYERS = 2
base_enc_layers = []
custom_enc_layers = []

enc_context_id = layer_cuda_module.create_global_context(True)

for _ in range(NUM_LAYERS):
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


@kt.case(dtypes=[torch.half], rtol=1e-3, atol=1e-2, ntest=5, nrepeat=5)
def test_encoder_layer_backward():
    batch_size, seq_len = kt.bs_sl()
    print(f"(batch_size, seq_len): ({batch_size}, {seq_len})")

    hidden_size = 1024

    hidden_states = kt.rand((batch_size, seq_len, hidden_size)).requires_grad_()
    self_attn_padding_mask = kt.attn_mask(batch_size, seq_len, dtype=torch.bool)
    loss_data = torch.randn(1, dtype=hidden_states.dtype).sum()

    def custom():
        for i in range(NUM_LAYERS):
            custom_enc_layers[i].zero_grad()
        res = hidden_states.clone()
        for i in range(NUM_LAYERS):
            res = custom_enc_layers[i](res, self_attn_padding_mask)
        custom_loss = (res / 1000).sum()
        custom_loss.data.copy_(loss_data)
        custom_loss.backward()
        grad_list = []
        for i in range(NUM_LAYERS - 1, -1, -1):
            """
            attn_qkvw, attn_qkvb, attn_ow, attn_ob, attn_nw, attn_nb,
            inter_w, inter_b, output_w, output_b, ffn_nw, ffn_nb
            """
            grads = split_custom_layer_grad(custom_enc_layers[i])
            grad_list.extend(grads[:11])
            pass
        return grad_list

    def baseline():
        for i in range(NUM_LAYERS):
            base_enc_layers[i].zero_grad()
        res = hidden_states.clone()
        for i in range(NUM_LAYERS):
            res = base_enc_layers[i](res, self_attn_padding_mask)
        custom_loss = (res / 1000).sum()
        custom_loss.data.copy_(loss_data)
        custom_loss.backward()
        grad_list = []
        for i in range(NUM_LAYERS - 1, -1, -1):
            """
            attn_qkvw, attn_qkvb, attn_ow, attn_ob, attn_nw, attn_nb,
            inter_w, inter_b, output_w, output_b, ffn_nw, ffn_nb
            """
            grads = split_custom_layer_grad(base_enc_layers[i])
            grad_list.extend(grads[:11])
            pass
        return grad_list

    return custom, baseline


###################### generate decoder layers ######################


def generate_dec_layer(initial_weights=None, initial_biases=None):
    config = LSFSTransformerDecoderLayerNew.get_config(
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
        nlayer=NUM_LAYERS,
        activation_fn="relu",
    )
    layer = LSFSTransformerDecoderLayerNew(
        config,
        initial_weights,
        initial_biases,
    )
    layer.to(torch.device("cuda:0"), dtype=torch.half)

    base_config = LSFSTransformerDecoderLayer.get_config(
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
        nlayer=NUM_LAYERS,
        activation_fn="relu",
    )
    base_layer = LSFSTransformerDecoderLayer(
        base_config,
        initial_weights,
        initial_biases,
    )
    base_layer.to(torch.device("cuda:0"), dtype=torch.half)
    return layer, base_layer


base_dec_layer_list = []
custom_dec_layer_list = []
fairseq_dec_layer_list = []
_initial_dec_weights_list = []
_initial_dec_biases_list = []
_initial_encdec_attn_kvw_list = []
_initial_encdec_attn_kvb_list = []

for _ in range(NUM_LAYERS):
    fairseq_dec_layer = fairseq_layers.generate_dec_layer()
    fairseq_dec_layer.train()
    initial_dec_weights, initial_dec_biases = get_fairseq_dec_params(fairseq_dec_layer)
    fairseq_dec_layer_list.append(fairseq_dec_layer)
    _initial_dec_weights_list.append(initial_dec_weights)
    _initial_dec_biases_list.append(initial_dec_biases)
    _initial_encdec_attn_kvw_list.append(initial_dec_weights[6])
    _initial_encdec_attn_kvw_list.append(initial_dec_weights[7])
    _initial_encdec_attn_kvb_list.append(initial_dec_biases[6])
    _initial_encdec_attn_kvb_list.append(initial_dec_biases[7])

_initial_encdec_attn_kvw = torch.cat(_initial_encdec_attn_kvw_list, dim=0)
_initial_encdec_attn_kvb = torch.cat(_initial_encdec_attn_kvb_list, dim=0)

dec_context_id = layer_cuda_module.create_global_context(True)

for i in range(NUM_LAYERS):
    _initial_dec_weights_list[i].pop(7)
    _initial_dec_weights_list[i].pop(6)
    if i == 0:
        _initial_dec_weights_list[i].append(_initial_encdec_attn_kvw)
    _initial_dec_biases_list[i].pop(7)
    _initial_dec_biases_list[i].pop(6)
    if i == 0:
        _initial_dec_biases_list[i].append(_initial_encdec_attn_kvb)
    custom_dec_layer, base_dec_layer = generate_dec_layer(
        _initial_dec_weights_list[i], _initial_dec_biases_list[i]
    )
    custom_dec_layer.train()
    base_dec_layer.train()
    custom_dec_layer_list.append(custom_dec_layer)
    base_dec_layer_list.append(base_dec_layer)


@kt.case(dtypes=[torch.half], rtol=1e-3, atol=1e-2, ntest=5, nrepeat=5)
def test_decoder_layer_forward():
    batch_size, enc_seq_len = kt.bs_sl()
    _, dec_seq_len = kt.bs_sl(batch_size)
    batch_size, enc_seq_len, dec_seq_len = 11, 428, 563
    print(
        f"(batch_size, enc_seq_len, dec_seq_len): ({batch_size}, {enc_seq_len},"
        f" {dec_seq_len})"
    )

    hidden_states = kt.rand((batch_size, dec_seq_len, 1024))
    encoder_out = kt.rand((enc_seq_len, batch_size, 1024))
    incremental_state = None
    encoder_padding_mask = kt.attn_mask(batch_size, enc_seq_len, dtype=torch.bool)
    self_attn_mask = kt.dec_self_attn_mask(dec_seq_len) * -1e8

    def custom():
        res = hidden_states.clone()
        enc_copy = encoder_out.clone()
        for i in range(NUM_LAYERS):
            res, _, _ = custom_dec_layer_list[i](
                res,
                encoder_out=enc_copy,
                encoder_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
            )
        return [
            res.contiguous().detach(),
        ]

    def baseline():
        res = hidden_states.clone()
        enc_copy = encoder_out.clone()
        for i in range(NUM_LAYERS):
            res, _, _ = base_dec_layer_list[i](
                res,
                encoder_out=enc_copy,
                encoder_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
            )
        return [
            res.contiguous().detach(),
        ]

    return custom, baseline


if __name__ == "__main__":

    kt.init(device="cuda:0", nhead=16)
    kt.run(
        [
            "test_encoder_layer_forward",
            "test_encoder_layer_backward",
            "test_decoder_layer_forward",
        ]
    )
