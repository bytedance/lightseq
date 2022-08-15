import random
import math
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn


from tests.util import TestDecorator
# from lightseq.training.ops.pytorch import util

from lightseq.csrc.pybind.builder import OperatorBuilder

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

    torch.cuda.set_stream(torch.cuda.Stream())
    print('--------->', torch.cuda.current_stream())

    def custom():
        res = hidden_states.clone()
        
        if kt.dtype == torch.float:
            func = op_module.layer_normalize_fw_fp32
        else:
            func = op_module.layer_normalize_fw_fp16

        func(ln_res, hidden_states, gamma, betta, hidden_dim, batch_size * seq_len)

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




if __name__ == "__main__":
    kt.init(device="cuda:0", nhead=16)
    kt.run(
        [
            "test_layer_normalize_fw",
        ]
    )
