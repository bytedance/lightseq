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
from lightseq.training.ops.pytorch.transformer_encoder_layer import (
    LSTransformerEncoderLayer,
)
from lightseq.training.ops.pytorch.transformer_embedding_layer import (
    LSTransformerEmbeddingLayer,
)
from lightseq.training.ops.pytorch.cross_entropy_layer import LSCrossEntropyLayer
from examples.training.fairseq.fs_modules.ls_fs_transformer_decoder_layer import (
    LSFSTransformerDecoderLayer,
)

kt = TestDecorator()

num_layers = 1

###################### encoding layer ######################


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
    )
    layer = LSTransformerEncoderLayer(config, initial_weights, initial_biases)
    layer.to(torch.device("cuda:0"), dtype=torch.half)
    return layer


custom_enc_layer_list = []
fairseq_enc_layer_list = []


def gen_enc_layer_pair():
    fairseq_enc_layer = fairseq_layers.generate_enc_layer()
    fairseq_enc_layer.train()
    initial_enc_weights, initial_enc_biases = get_fairseq_enc_params(fairseq_enc_layer)
    custom_enc_layer = generate_enc_layer(initial_enc_weights, initial_enc_biases)
    custom_enc_layer.train()
    return fairseq_enc_layer, custom_enc_layer


for _ in range(num_layers):
    fairseq_enc_layer, custom_enc_layer = gen_enc_layer_pair()
    custom_enc_layer_list.append(custom_enc_layer)
    fairseq_enc_layer_list.append(fairseq_enc_layer)

###################### decoding layer ######################


def generate_dec_layer(initial_weights=None, initial_biases=None):
    config = LSFSTransformerDecoderLayer.get_config(
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
        nlayer=num_layers,
    )
    layer = LSFSTransformerDecoderLayer(
        config,
        initial_weights,
        initial_biases,
    )
    layer.to(torch.device("cuda:0"), dtype=torch.half)
    return layer


custom_dec_layer_list = []
fairseq_dec_layer_list = []
_initial_dec_weights_list = []
_initial_dec_biases_list = []
_initial_encdec_attn_kvw_list = []
_initial_encdec_attn_kvb_list = []

for _ in range(num_layers):
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
for i in range(num_layers):
    _initial_dec_weights_list[i].pop(7)
    _initial_dec_weights_list[i].pop(6)
    if i == 0:
        _initial_dec_weights_list[i].append(_initial_encdec_attn_kvw)
    _initial_dec_biases_list[i].pop(7)
    _initial_dec_biases_list[i].pop(6)
    if i == 0:
        _initial_dec_biases_list[i].append(_initial_encdec_attn_kvb)
    custom_dec_layer = generate_dec_layer(
        _initial_dec_weights_list[i], _initial_dec_biases_list[i]
    )
    custom_dec_layer.train()
    custom_dec_layer_list.append(custom_dec_layer)

###################### embedding layer ######################

ls_emb_config_fp16 = LSTransformerEmbeddingLayer.get_config(
    vocab_size=40480,
    embedding_dim=1024,
    max_batch_tokens=9216,
    max_seq_len=256,
    padding_idx=2,
    dropout=0.0,
    fp16=True,
    local_rank=0,
)
ls_emb_config_fp32 = deepcopy(ls_emb_config_fp16)
ls_emb_config_fp32.fp16 = False

fs_emb_layer_fp32 = fairseq_layers.generate_emb_layer(ls_emb_config_fp32)
fs_emb_layer_fp16 = fairseq_layers.generate_emb_layer(ls_emb_config_fp16)
fs_emb_layer_fp32.train()
fs_emb_layer_fp16.train()


def generate_emb_layer(config, initial_weights=None):
    custom_layer = LSTransformerEmbeddingLayer(config, initial_weights)
    dtype = torch.float16 if config.fp16 else torch.float32
    custom_layer.to(torch.device("cuda:0"), dtype=dtype)
    return custom_layer


custom_emb_layer_fp32 = generate_emb_layer(
    ls_emb_config_fp32, fs_emb_layer_fp32.embeddings.weight.detach().clone()
)
custom_emb_layer_fp16 = generate_emb_layer(
    ls_emb_config_fp16, fs_emb_layer_fp16.embeddings.weight.detach().clone()
)
custom_emb_layer_fp32.train()
custom_emb_layer_fp16.train()

###################### cross entropy layer ######################

ce_config_fp16 = LSCrossEntropyLayer.get_config(
    max_batch_tokens=9216,
    padding_idx=2,
    epsilon=0.1,
    fp16=True,
    local_rank=0,
)
ce_config_fp32 = deepcopy(ce_config_fp16)
ce_config_fp32.fp16 = False


def generate_cross_entropy_layer(config):
    dtype = torch.float16 if config.fp16 else torch.float32
    custom_layer = LSCrossEntropyLayer(config)
    custom_layer.to(torch.device("cuda:0"), dtype=dtype)
    return custom_layer


custom_cross_entropy_layer_fp32 = generate_cross_entropy_layer(ce_config_fp32)
custom_cross_entropy_layer_fp16 = generate_cross_entropy_layer(ce_config_fp16)
custom_cross_entropy_layer_fp32.train()
custom_cross_entropy_layer_fp16.train()


@kt.case(dtypes=[torch.half], rtol=1e-3, atol=1e-2, ntest=10)
def test_encoder_layer_forward():
    batch_size, seq_len = kt.bs_sl()
    print(f"(batch_size, seq_len): ({batch_size}, {seq_len})")

    hidden_states = kt.rand((batch_size, seq_len, 1024))
    self_attn_padding_mask = kt.attn_mask(batch_size, seq_len, dtype=torch.bool)

    def custom():
        res = hidden_states.clone()
        for i in range(num_layers):
            res = custom_enc_layer_list[i](res, self_attn_padding_mask)
        return [
            res.contiguous().detach(),
        ]

    def baseline():
        res = hidden_states.transpose(0, 1).contiguous().clone()
        for i in range(num_layers):
            res = fairseq_enc_layer_list[i](res, self_attn_padding_mask)
        return [
            res.transpose(0, 1).contiguous().detach(),
        ]

    return custom, baseline


@kt.case(dtypes=[torch.half], rtol=1e-2, atol=1e-2, ntest=10)
def test_encoder_layer_backward():
    batch_size, seq_len = kt.bs_sl()
    print(f"(batch_size, seq_len): ({batch_size}, {seq_len})")
    hidden_size = 1024
    shs = hidden_size * hidden_size

    hidden_states = kt.rand((batch_size, seq_len, hidden_size))
    self_attn_padding_mask = kt.attn_mask(batch_size, seq_len, dtype=torch.bool)
    loss_data = torch.randn(1, dtype=hidden_states.dtype).sum()

    def custom():
        for i in range(num_layers):
            custom_enc_layer_list[i].zero_grad()
        res = hidden_states.clone()
        for i in range(num_layers):
            res = custom_enc_layer_list[i](res, self_attn_padding_mask)
        custom_loss = (res / 1000).sum()
        custom_loss.data.copy_(loss_data)
        custom_loss.backward()
        grad_list = []
        for i in range(num_layers - 1, -1, -1):
            """
            attn_qkvw, attn_qkvb, attn_ow, attn_ob, attn_nw, attn_nb,
            inter_w, inter_b, output_w, output_b, ffn_nw, ffn_nb
            """
            grads = split_custom_layer_grad(custom_enc_layer_list[i])
            grad_list.extend(
                [
                    grads[8],
                    grads[9],
                    grads[6],
                    grads[7],
                    grads[10],
                    grads[11],
                    grads[2],
                    grads[3],
                    grads[0][:shs],
                    grads[1][:hidden_size],
                    grads[0][shs : shs * 2],
                    grads[1][hidden_size : hidden_size * 2],
                    grads[0][shs * 2 : shs * 3],
                    grads[1][hidden_size * 2 : hidden_size * 3],
                    grads[4],
                    grads[5],
                ]
            )
        return grad_list

    def baseline():
        for i in range(num_layers):
            fairseq_enc_layer_list[i].zero_grad()
        res = hidden_states.transpose(0, 1).clone()
        for i in range(num_layers):
            res = fairseq_enc_layer_list[i](res, self_attn_padding_mask)
        fairseq_loss = (res / 1000).sum()
        fairseq_loss.data.copy_(loss_data)
        fairseq_loss.backward()
        grad_list = []
        for i in range(num_layers - 1, -1, -1):
            curl = fairseq_enc_layer_list[i]
            cur_grads = copy_grad_from_paras(
                [
                    curl.fc2.weight,
                    curl.fc2.bias,
                    curl.fc1.weight,
                    curl.fc1.bias,
                    curl.final_layer_norm.weight,
                    curl.final_layer_norm.bias,
                    curl.self_attn.out_proj.weight,
                    curl.self_attn.out_proj.bias,
                    curl.self_attn.q_proj.weight,
                    curl.self_attn.q_proj.bias,
                    curl.self_attn.k_proj.weight,
                    curl.self_attn.k_proj.bias,
                    curl.self_attn.v_proj.weight,
                    curl.self_attn.v_proj.bias,
                    curl.self_attn_layer_norm.weight,
                    curl.self_attn_layer_norm.bias,
                ]
            )
            grad_list.extend(cur_grads)
        return grad_list

    return custom, baseline


@kt.case(dtypes=[torch.half], rtol=1e-3, atol=1e-2, ntest=10)
def test_decoder_layer_forward():
    batch_size, enc_seq_len = kt.bs_sl()
    _, dec_seq_len = kt.bs_sl(batch_size)
    print(
        f"(batch_size, enc_seq_len, dec_seq_len): ({batch_size}, {enc_seq_len}, {dec_seq_len})"
    )

    hidden_states = kt.rand((batch_size, dec_seq_len, 1024))
    encoder_out = kt.rand((enc_seq_len, batch_size, 1024))
    incremental_state = None
    encoder_padding_mask = kt.attn_mask(batch_size, enc_seq_len, dtype=torch.bool)
    self_attn_mask = kt.dec_self_attn_mask(dec_seq_len) * -1e8

    def custom():
        res = hidden_states.clone()
        for i in range(num_layers):
            res, _, _ = custom_dec_layer_list[i](
                res,
                encoder_out=encoder_out,
                encoder_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
            )
        return [
            res.contiguous().detach(),
        ]

    def baseline():
        res = hidden_states.transpose(0, 1).clone()
        for i in range(num_layers):
            res, _, _ = fairseq_dec_layer_list[i](
                res,
                encoder_out=encoder_out,
                encoder_padding_mask=encoder_padding_mask,
                self_attn_mask=self_attn_mask,
                incremental_state=incremental_state,
            )
        return [
            res.transpose(0, 1).contiguous().detach(),
        ]

    return custom, baseline


@kt.case(dtypes=[torch.half], rtol=1e-2, atol=1e-2, ntest=10)
def test_decoder_layer_backward():
    batch_size, enc_seq_len = kt.bs_sl()
    _, dec_seq_len = kt.bs_sl(batch_size)
    print(
        f"(batch_size, enc_seq_len, dec_seq_len): ({batch_size}, {enc_seq_len}, {dec_seq_len})"
    )
    hidden_size = 1024
    shs = hidden_size * hidden_size
    hidden_states = kt.rand((batch_size, dec_seq_len, hidden_size))
    encoder_out = kt.rand((enc_seq_len, batch_size, hidden_size))
    incremental_state = None
    encoder_padding_mask = kt.attn_mask(batch_size, enc_seq_len, dtype=torch.bool)
    self_attn_mask = kt.dec_self_attn_mask(dec_seq_len) * -1e8
    loss_data = torch.randn(1, dtype=hidden_states.dtype).sum()

    def custom():
        for i in range(num_layers):
            custom_dec_layer_list[i].zero_grad()
        res = hidden_states.clone()
        for i in range(num_layers):
            res, _, _ = custom_dec_layer_list[i](
                res,
                encoder_out=encoder_out,
                encoder_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
            )
        custom_loss = (res / 1000).sum()
        custom_loss.data.copy_(loss_data)
        custom_loss.backward()
        grad_list = []
        for i in range(num_layers - 1, -1, -1):
            """
            0 attn_qkvw, attn_qkvb, attn_ow, attn_ob, attn_nw, attn_nb,
            6 encdec_attn_qw, encdec_attn_qb, encdec_attn_ow, encdec_attn_ob, encdec_attn_nw, encdec_attn_nb,
            12 inter_w, inter_b, output_w, output_b, ffn_nw, ffn_nb
            18 encdec_attn_kvw, encdec_attn_kvb,
            """
            grads = split_custom_layer_grad(custom_dec_layer_list[i])
            grad_list.extend(
                [
                    grads[14],
                    grads[15],
                    grads[12],
                    grads[13],
                    grads[16],
                    grads[17],
                    grads[2],
                    grads[3],
                    grads[0][:shs],
                    grads[1][:hidden_size],
                    grads[0][shs : shs * 2],
                    grads[1][hidden_size : hidden_size * 2],
                    grads[0][shs * 2 : shs * 3],
                    grads[1][hidden_size * 2 : hidden_size * 3],
                    grads[4],
                    grads[5],
                    # encdec grad
                    grads[6],
                    grads[7],
                    grads[8],
                    grads[9],
                    grads[10],
                    grads[11],
                ]
            )
            if i == 0:
                grad_list.extend(
                    [
                        # encdec kv grad
                        grads[18][:shs],
                        grads[19][:hidden_size],
                        grads[18][shs : shs * 2],
                        grads[19][hidden_size : hidden_size * 2],
                    ]
                )
        return grad_list

    def baseline():
        for i in range(num_layers):
            fairseq_dec_layer_list[i].zero_grad()
        res = hidden_states.transpose(0, 1).clone()
        for i in range(num_layers):
            res, _, _ = fairseq_dec_layer_list[i](
                res,
                encoder_out=encoder_out,
                encoder_padding_mask=encoder_padding_mask,
                self_attn_mask=self_attn_mask,
                incremental_state=incremental_state,
            )
        fairseq_loss = (res / 1000).sum()
        fairseq_loss.data.copy_(loss_data)
        fairseq_loss.backward()
        grad_list = []
        for i in range(num_layers - 1, -1, -1):
            grad_list.extend(
                [
                    fairseq_dec_layer_list[i].fc2.weight.grad.contiguous().detach(),
                    fairseq_dec_layer_list[i].fc2.bias.grad.contiguous().detach(),
                    fairseq_dec_layer_list[i].fc1.weight.grad.contiguous().detach(),
                    fairseq_dec_layer_list[i].fc1.bias.grad.contiguous().detach(),
                    fairseq_dec_layer_list[i]
                    .final_layer_norm.weight.grad.contiguous()
                    .detach(),
                    fairseq_dec_layer_list[i]
                    .final_layer_norm.bias.grad.contiguous()
                    .detach(),
                    fairseq_dec_layer_list[i]
                    .self_attn.out_proj.weight.grad.contiguous()
                    .detach(),
                    fairseq_dec_layer_list[i]
                    .self_attn.out_proj.bias.grad.contiguous()
                    .detach(),
                    fairseq_dec_layer_list[i]
                    .self_attn.q_proj.weight.grad.contiguous()
                    .detach(),
                    fairseq_dec_layer_list[i]
                    .self_attn.q_proj.bias.grad.contiguous()
                    .detach(),
                    fairseq_dec_layer_list[i]
                    .self_attn.k_proj.weight.grad.contiguous()
                    .detach(),
                    fairseq_dec_layer_list[i]
                    .self_attn.k_proj.bias.grad.contiguous()
                    .detach(),
                    fairseq_dec_layer_list[i]
                    .self_attn.v_proj.weight.grad.contiguous()
                    .detach(),
                    fairseq_dec_layer_list[i]
                    .self_attn.v_proj.bias.grad.contiguous()
                    .detach(),
                    fairseq_dec_layer_list[i]
                    .self_attn_layer_norm.weight.grad.contiguous()
                    .detach(),
                    fairseq_dec_layer_list[i]
                    .self_attn_layer_norm.bias.grad.contiguous()
                    .detach(),
                    # encdec weights grad
                    fairseq_dec_layer_list[i]
                    .encodec_attn.q_proj.weight.grad.contiguous()
                    .detach(),
                    fairseq_dec_layer_list[i]
                    .encodec_attn.q_proj.bias.grad.contiguous()
                    .detach(),
                    fairseq_dec_layer_list[i]
                    .encodec_attn.out_proj.weight.grad.contiguous()
                    .detach(),
                    fairseq_dec_layer_list[i]
                    .encodec_attn.out_proj.bias.grad.contiguous()
                    .detach(),
                    fairseq_dec_layer_list[i]
                    .encodec_attn_layer_norm.weight.grad.contiguous()
                    .detach(),
                    fairseq_dec_layer_list[i]
                    .encodec_attn_layer_norm.bias.grad.contiguous()
                    .detach(),
                ]
            )
            if i == 0:
                grad_list.extend(
                    [
                        # encdec kv grad
                        fairseq_dec_layer_list[i]
                        .encodec_attn.k_proj.weight.grad.contiguous()
                        .detach(),
                        fairseq_dec_layer_list[i]
                        .encodec_attn.k_proj.bias.grad.contiguous()
                        .detach(),
                        fairseq_dec_layer_list[i]
                        .encodec_attn.v_proj.weight.grad.contiguous()
                        .detach(),
                        fairseq_dec_layer_list[i]
                        .encodec_attn.v_proj.bias.grad.contiguous()
                        .detach(),
                    ]
                )
        return grad_list

    return custom, baseline


@kt.case(dtypes=[torch.half], rtol=1e-3, atol=1e-2, ntest=10, nrepeat=1)
def test_decoder_layer_forward_inference():
    batch_size, enc_seq_len = kt.bs_sl()
    print(f"(batch_size, enc_seq_len): ({batch_size}, {enc_seq_len})")
    # beam_size = random.randint(2,5)
    # print(f"(batch_size, enc_seq_len, beam_size): ({batch_size}, {enc_seq_len}, {beam_size})")

    # ls_encoder_out = kt.rand((batch_size, enc_seq_len, 1024))
    # fs_encoder_out = ls_encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).reshape(-1, enc_seq_len, 1024)
    # ls_enc_mask = kt.attn_mask(batch_size, enc_seq_len, dtype=torch.bool)
    # fs_enc_mask = ls_enc_mask.unsqueeze(1).repeat(1, beam_size, 1).reshape(-1, enc_seq_len)
    encoder_out = kt.rand((enc_seq_len, batch_size, 1024))
    encoder_padding_mask = kt.attn_mask(batch_size, enc_seq_len, dtype=torch.bool)

    hidden_states_list = []
    max_step = 10
    for i in range(max_step):
        # hidden_states = kt.rand((batch_size * beam_size, 1, 1024))
        hidden_states = kt.rand((batch_size, 1, 1024))
        hidden_states_list.append(hidden_states)

    def custom():
        incremental_state = {}
        res_list = []
        for i in range(max_step):
            res = hidden_states_list[i].clone()
            for i in range(num_layers):
                res, _, _ = custom_dec_layer_list[i](
                    res,
                    # encoder_out=ls_encoder_out.transpose(0,1),
                    # encoder_padding_mask=ls_enc_mask,
                    encoder_out=encoder_out,
                    encoder_padding_mask=encoder_padding_mask,
                    incremental_state=incremental_state,
                )
            res_list.append(res)
        return [x.contiguous().detach() for x in res_list]

    def baseline():
        incremental_state = {}
        res_list = []
        for i in range(max_step):
            res = hidden_states_list[i].transpose(0, 1).clone()
            for i in range(num_layers):
                res, _, _ = fairseq_dec_layer_list[i](
                    res,
                    encoder_out=encoder_out,
                    encoder_padding_mask=encoder_padding_mask,
                    incremental_state=incremental_state,
                )
            res_list.append(res)
        return [x.transpose(0, 1).contiguous().detach() for x in res_list]

    return custom, baseline


@kt.case(ntest=10)
def test_embedding_layer_forward():
    batch_size, seq_len = kt.bs_sl()
    print(f"(batch_size, seq_len): ({batch_size}, {seq_len})")

    padding_mask = kt.attn_mask(batch_size, seq_len, dtype=torch.int)
    # TODO: can not generate PAD in the middle of the sentences.
    config = ls_emb_config_fp16
    input = kt.randint(config.padding_idx + 1, config.vocab_size, (batch_size, seq_len))
    input = input * (1 - padding_mask) + config.padding_idx * padding_mask

    if kt.dtype == torch.float:
        custom_layer = custom_emb_layer_fp32
        fs_layer = fs_emb_layer_fp32
    else:
        custom_layer = custom_emb_layer_fp16
        fs_layer = fs_emb_layer_fp16

    def custom():
        res = custom_layer(input)
        return [
            res.contiguous().detach(),
        ]

    def baseline():
        x = fs_layer(input)
        return [
            x.contiguous().detach(),
        ]

    return custom, baseline


@kt.case(ntest=10)
def test_embedding_layer_backward():
    batch_size, seq_len = kt.bs_sl()
    print(f"(batch_size, seq_len): ({batch_size}, {seq_len})")

    padding_mask = kt.attn_mask(batch_size, seq_len, dtype=torch.int)
    config = ls_emb_config_fp16
    input = kt.randint(config.padding_idx + 1, config.vocab_size, (batch_size, seq_len))
    input = input * (1 - padding_mask) + config.padding_idx * padding_mask

    if kt.dtype == torch.float:
        custom_layer = custom_emb_layer_fp32
        fs_layer = fs_emb_layer_fp32
    else:
        custom_layer = custom_emb_layer_fp16
        fs_layer = fs_emb_layer_fp16

    loss_data = torch.randn(1, dtype=kt.dtype).sum()

    def custom():
        custom_layer.zero_grad()
        custom_input = input.clone()
        res = custom_layer(custom_input)
        custom_loss = (res / 1000).sum()
        custom_loss.data.copy_(loss_data)
        custom_loss.backward()
        return [
            custom_layer.embeddings.grad.contiguous().detach(),
        ]

    def baseline():
        fs_layer.zero_grad()
        fs_input = input.clone()
        res = fs_layer(fs_input)
        fs_loss = (res / 1000).sum()
        fs_loss.data.copy_(loss_data)
        fs_loss.backward()
        return [
            fs_layer.embeddings.weight.grad.contiguous().detach(),
        ]

    return custom, baseline


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@kt.case(ntest=10)
def test_cross_entropy_layer_forward():
    batch_size, seq_len = kt.bs_sl()
    vocab_size = random.randint(30413, 40519)
    print(f"(batch_size, seq_len): ({batch_size}, {seq_len})")

    inputs = kt.rand((batch_size, seq_len, vocab_size))
    targets = kt.randint(
        ce_config_fp16.padding_idx - 1, vocab_size, (batch_size, seq_len)
    )
    targets_32 = targets.to(torch.int32)

    if kt.dtype == torch.float:
        custom_layer = custom_cross_entropy_layer_fp32
    else:
        custom_layer = custom_cross_entropy_layer_fp16

    def custom():
        res, cus_nll_loss = custom_layer(inputs, targets_32)
        return [
            res.contiguous().detach(),
            cus_nll_loss.contiguous().detach(),
        ]

    def baseline():

        x = torch.nn.functional.log_softmax(inputs, dim=-1, dtype=torch.float32)
        x, base_nll_loss = label_smoothed_nll_loss(
            x, targets, ce_config_fp16.epsilon, ignore_index=ce_config_fp16.padding_idx
        )
        return [
            x.contiguous().detach(),
            base_nll_loss.contiguous().detach(),
        ]

    return custom, baseline


@kt.case(ntest=10)
def test_cross_entropy_layer_backward():
    batch_size, seq_len = kt.bs_sl()
    vocab_size = random.randint(30413, 40519)
    print(f"(batch_size, seq_len): ({batch_size}, {seq_len})")

    base_inputs = kt.rand((batch_size, seq_len, vocab_size)).requires_grad_()
    cus_inputs = base_inputs.clone().detach().requires_grad_()
    targets = kt.randint(
        ce_config_fp16.padding_idx - 1, vocab_size, (batch_size, seq_len)
    )
    targets_32 = targets.to(torch.int32)

    if kt.dtype == torch.float:
        custom_layer = custom_cross_entropy_layer_fp32
    else:
        custom_layer = custom_cross_entropy_layer_fp16
    cus_res = custom_layer(cus_inputs, targets_32)[0].to(kt.dtype)
    x = torch.nn.functional.log_softmax(base_inputs, dim=-1, dtype=torch.float32)
    base_res, _ = label_smoothed_nll_loss(
        x, targets, ce_config_fp16.epsilon, ignore_index=ce_config_fp16.padding_idx
    )
    base_res = base_res.to(kt.dtype)

    def custom():
        if cus_inputs.grad is not None:
            cus_inputs.grad.zero_()
        cus_res.backward(retain_graph=True)
        return [
            cus_inputs.grad.contiguous().detach(),
        ]

    def baseline():
        if base_inputs.grad is not None:
            base_inputs.grad.zero_()
        base_res.backward(retain_graph=True)
        # print(base_inputs.grad)
        return [
            base_inputs.grad.contiguous().detach(),
        ]

    return custom, baseline


if __name__ == "__main__":
    kt.init(device="cuda:0", nhead=16)
    kt.run(
        [
            "test_encoder_layer_forward",
            "test_encoder_layer_backward",
            "test_decoder_layer_forward",
            "test_decoder_layer_backward",
            "test_decoder_layer_forward_inference",
            "test_embedding_layer_forward",
            "test_embedding_layer_backward",
            "test_cross_entropy_layer_forward",
            "test_cross_entropy_layer_backward",
        ]
    )
