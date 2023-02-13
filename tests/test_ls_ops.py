import random
import math
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn

from tests.util import TestDecorator
from lightseq.training.ops.pytorch import OperatorBuilder

kt = TestDecorator()
op_module = OperatorBuilder().load()


@kt.case()
def test_split_head_op():
    batch_size, seq_len = kt.bs_sl()
    hidden_dim = kt.hidden_dim
    nhead = kt.nhead
    head_dim = int(hidden_dim / nhead)
    qkv_num = random.choice([1, 3])

    print(
        "(batch_size, seq_len, qkv_num, nhead, head_dim): "
        f"({batch_size}, {seq_len}, {qkv_num}, {nhead}, {head_dim})"
    )

    input = kt.rand((batch_size, seq_len, qkv_num, hidden_dim))
    bias = kt.rand((hidden_dim))
    query = kt.rand((batch_size, seq_len, hidden_dim))
    key = kt.rand((batch_size, seq_len, hidden_dim))
    value = kt.rand((batch_size, seq_len, hidden_dim))

    if kt.dtype == torch.float:
        func = op_module.torch_split_head_op_fp32
    else:
        func = op_module.torch_split_head_op_fp16

    def custom():
        func(
            input,
            bias,
            query,
            key,
            value,
            batch_size,
            hidden_dim,
            nhead,
            seq_len,
            qkv_num,
        )
        if qkv_num == 3:
            return kt.norm_res_list(query, key, value)
        else:
            return kt.norm_res_list(query)

    def baseline():
        q, k, v = None, None, None
        func = lambda x: x.reshape((batch_size, seq_len, nhead, head_dim)).permute(
            0, 2, 1, 3
        )
        inp = input + bias
        if qkv_num == 3:
            q, k, v = inp.split(1, dim=2)
            q = func(q.squeeze())
            k = func(k.squeeze())
            v = func(v.squeeze())
            return kt.norm_res_list(query, key, value)
        else:
            q = func(inp.squeeze())
            return kt.norm_res_list(query)

    return custom, baseline


if __name__ == "__main__":
    kt.init(device="cuda:0", nhead=16)
    kt.run(
        [
            "test_split_head_op",
        ]
    )
