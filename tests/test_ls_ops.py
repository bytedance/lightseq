import random
import math
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn


from tests.util import TestDecorator, copy_grad_from_paras

# from lightseq.training.ops.pytorch import util

from lightseq.training.ops.pytorch import OperatorBuilder

kt = TestDecorator()
op_module = OperatorBuilder().load()

# def copy_para(x):
#     return torch.nn.Parameter(torch.empty_like(x).copy_(x))


def copy_para(x, fp16):
    y = torch.nn.Parameter(torch.empty_like(x).copy_(x)).to(x.device)
    return y.half() if fp16 else y.float()


@kt.case(dtypes=[torch.float], rtol=1e-3, atol=1e-2, ntest=10)
def test_layer_normalize_fw():
    batch_size, seq_len = kt.bs_sl()
    hidden_dim = 512

    hidden_states = kt.rand((batch_size, seq_len, hidden_dim))
    ln_res = torch.empty_like(hidden_states)
    gamma = kt.rand((hidden_dim))
    betta = kt.rand((hidden_dim))

    # torch.cuda.set_stream(torch.cuda.Stream())

    def custom():
        res = hidden_states.clone()

        if kt.dtype == torch.float:
            func = op_module.layer_normalize_fw_fp32
        else:
            func = op_module.layer_normalize_fw_fp16

        func(ln_res, res, gamma, betta, hidden_dim, batch_size * seq_len)

        return [
            ln_res.contiguous().detach(),
        ]

    def baseline():
        layer_norm = nn.LayerNorm(hidden_dim).to("cuda:0")
        layer_norm.weight.data.copy_(copy_para(gamma, kt.dtype))
        layer_norm.bias.data.copy_(copy_para(betta, kt.dtype))

        baseline_out = layer_norm(hidden_states)
        return [
            baseline_out.contiguous().detach(),
        ]

    return custom, baseline


@kt.case(dtypes=[torch.float], rtol=1e-3, atol=1e-2, ntest=10)
def test_layer_normalize_bw():
    batch_size, seq_len = kt.bs_sl()
    hidden_dim = 512

    hidden_states = kt.rand((batch_size, seq_len, hidden_dim))
    ln_res = torch.empty_like(hidden_states)
    gamma = kt.rand((hidden_dim))
    betta = kt.rand((hidden_dim))
    loss_data = torch.randn(1, dtype=hidden_states.dtype).sum()

    def custom():
        ln_res_grad = torch.empty_like(ln_res)
        res = hidden_states.clone()
        res_grad = torch.empty_like(res)
        gamma_grad = torch.empty_like(gamma)
        betta_grad = torch.empty_like(betta)

        if kt.dtype == torch.float:
            func = op_module.layer_normalize_bw_fp32
        else:
            func = op_module.layer_normalize_bw_fp16

        func(
            ln_res,
            ln_res_grad,
            res,
            res_grad,
            gamma,
            gamma_grad,
            betta,
            betta_grad,
            hidden_dim,
            batch_size * seq_len,
        )

        return [
            ln_res.contiguous().detach(),
            res_grad.contiguous().detach(),
            gamma_grad.contiguous().detach(),
            betta_grad.contiguous().detach(),
        ]

    def baseline():
        layer_norm = nn.LayerNorm(hidden_dim).to("cuda:0")
        layer_norm.weight.data.copy_(copy_para(gamma, kt.dtype))
        layer_norm.bias.data.copy_(copy_para(betta, kt.dtype))

        baseline_out = layer_norm(hidden_states)
        layer_norm.zero_grad()
        baseline_loss = (baseline_out / 1000.0).sum()
        baseline_loss.data_.copy(loss_data)
        baseline_loss.backward()

        all_tensor = [
            baseline_out.contiguous().detach(),
        ]
        grads = copy_grad_from_paras(
            [hidden_states, layer_norm.weight, layer_norm.bias]
        )
        all_tensor.extend(grads)
        return all_tensor

    return custom, baseline


@kt.case(dtypes=[torch.float], rtol=1e-3, atol=1e-2, ntest=10)
def test_feedforward_fw():
    batch_size, seq_len = kt.bs_sl()
    input_dim = 512
    output_dim = 768

    hidden_states = kt.rand((batch_size, seq_len, input_dim))
    weights = kt.rand((output_dim, input_dim))

    def custom():
        res = kt.zeros((batch_size, seq_len, output_dim))

        if kt.dtype == torch.float:
            func = op_module.feed_forward_fw_fp32
        else:
            func = op_module.feed_forward_fw_fp16

        func(
            hidden_states,
            weights.clone(),
            res,
            output_dim,
            input_dim,
            batch_size * seq_len,
        )

        return [
            res.contiguous().detach(),
        ]

    def baseline():
        feed_forward = nn.Linear(input_dim, output_dim, bias=False).to("cuda:0")
        feed_forward.weight.data.copy_(copy_para(weights, kt.dtype))
        baseline_out = feed_forward(hidden_states)

        return [
            baseline_out.contiguous().detach(),
        ]

    return custom, baseline


if __name__ == "__main__":
    kt.init(device="cuda:0", nhead=16)
    kt.run(
        [
            "test_layer_normalize_fw",
            # "test_layer_normalize_bw",
            "test_feedforward_fw",
        ]
    )
