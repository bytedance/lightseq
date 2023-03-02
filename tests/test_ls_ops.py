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
    bias = kt.rand((1, 1, qkv_num, hidden_dim))
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


@kt.case()
def test_split_head_with_beam_op():
    batch_size, q_len = kt.bs_sl()
    hidden_dim = kt.hidden_dim
    nhead = kt.nhead
    head_dim = int(hidden_dim / nhead)
    beam_size = random.randint(1, 5)
    cache_len = random.randint(q_len, q_len + 30)
    step = random.choice([0, 1])
    if step == 1:
        step = random.randint(0, cache_len - q_len)

    print(
        "(batch_size, q_len, step, beam_size): "
        f"({batch_size}, {q_len}, {step}, {beam_size})"
    )

    query_beam = 1 if step == 0 else beam_size

    input = kt.rand((query_beam, batch_size, q_len, 3, hidden_dim))
    bias = kt.rand((1, 1, 1, 3, hidden_dim))
    query = kt.rand((query_beam, batch_size, nhead, q_len, head_dim))
    key_cus = kt.rand((beam_size, batch_size, nhead, cache_len, head_dim))
    value_cus = kt.rand((beam_size, batch_size, nhead, cache_len, head_dim))
    key_baseline = key_cus.clone()
    value_baseline = value_cus.clone()

    if kt.dtype == torch.float:
        func = op_module.torch_split_head_with_beam_op_fp32
    else:
        func = op_module.torch_split_head_with_beam_op_fp16

    def custom():
        func(
            input,
            bias,
            query,
            key_cus,
            value_cus,
            batch_size,
            hidden_dim,
            nhead,
            beam_size,
            q_len,
            cache_len,
            step,
        )
        return kt.norm_res_list(query, key_cus, value_cus)

    def baseline():
        # [query_beam, batch_size, nhead, q_len, head_dim]
        func = (
            lambda x: x.squeeze()
            .reshape((query_beam, batch_size, q_len, nhead, head_dim))
            .permute(0, 1, 3, 2, 4)
        )
        inp = input + bias
        q, k, v = inp.split(1, dim=3)
        q = func(q)
        k = func(k)
        v = func(v)
        if step == 0:
            key_baseline[0:1, :, :, step : step + q_len, :] = k
            value_baseline[0:1, :, :, step : step + q_len, :] = v
        else:
            key_baseline[:, :, :, step : step + q_len, :] = k
            value_baseline[:, :, :, step : step + q_len, :] = v
        return kt.norm_res_list(q, key_baseline, value_baseline)

    return custom, baseline


if __name__ == "__main__":
    kt.init(device="cuda:0", nhead=16)
    kt.run(
        "test_split_head_op",
        "test_split_head_with_beam_op",
    )
