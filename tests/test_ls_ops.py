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
    batch_size, q_len = kt.bs_sl()
    hidden_dim = kt.hidden_dim
    nhead = kt.nhead
    head_dim = int(hidden_dim / nhead)
    qkv_num = random.choice([1, 3])

    step = 0
    kv_len = q_len
    cache_sz = random.choice([0, 1])
    if cache_sz == 1:
        cache_sz = random.randint(q_len, q_len + 30)
        step = random.randint(0, cache_sz - q_len)
        kv_len = cache_sz
        qkv_num = 3

    print(
        "(batch_size, q_len, qkv_num, cache_sz, step): "
        f"({batch_size}, {q_len}, {qkv_num}, {cache_sz}, {step})"
    )

    input = kt.rand((batch_size, q_len, qkv_num, hidden_dim))
    bias = kt.rand((1, 1, qkv_num, hidden_dim))
    query_cus = kt.rand((batch_size, nhead, q_len, head_dim))
    key_cus = kt.rand((batch_size, nhead, kv_len, head_dim))
    value_cus = kt.rand((batch_size, nhead, kv_len, head_dim))
    key_base = key_cus.clone()
    value_base = value_cus.clone()

    if kt.dtype == torch.float:
        func = op_module.torch_split_head_op_fp32
    else:
        func = op_module.torch_split_head_op_fp16

    def custom():
        func(
            input,
            bias,
            query_cus,
            key_cus,
            value_cus,
            batch_size,
            hidden_dim,
            nhead,
            q_len,
            qkv_num,
            cache_sz,
            step,
        )
        if qkv_num == 3:
            return kt.norm_res_list(query_cus, key_cus, value_cus)
        else:
            return kt.norm_res_list(query_cus)

    def baseline():
        q, k, v = None, None, None
        func = lambda x: x.reshape((batch_size, q_len, nhead, head_dim)).permute(
            0, 2, 1, 3
        )
        inp = input + bias
        if qkv_num == 3:
            q, k, v = inp.split(1, dim=2)
            q = func(q.squeeze())
            key_base[:, :, step : step + q_len, :] = func(k.squeeze())
            value_base[:, :, step : step + q_len, :] = func(v.squeeze())
            return kt.norm_res_list(q, key_base, value_base)
        else:
            q = func(inp.squeeze())
            return kt.norm_res_list(q)

    return custom, baseline


if __name__ == "__main__":
    kt.init(device="cuda:0", nhead=16)
    kt.run(
        "test_split_head_op",
    )