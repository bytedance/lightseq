import __init__
import random
import math
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch.nn import functional

from csrc.tests.util import TestDecorator
from csrc.pytorch.builder.cuda_layer_builder import CudaLayerBuilder

kt = TestDecorator()
layer_module = CudaLayerBuilder().load()


@kt.case(atol=1e-3, rtol=1e-3)
def test_sdpa_layer():
    batch_size, q_len = kt.bs_sl()
    hidden_dim = kt.hidden_dim
    nhead = kt.nhead
    batch_head = batch_size * nhead
    head_dim = int(hidden_dim / nhead)
    max_seq_len = random.randint(q_len, q_len + 10)
    max_batch_size = random.randint(batch_size, batch_size + 10)
    max_batch_tokens = max_seq_len * max_batch_size

    mask_future = None

    q_kv_same_len = random.choice([0, 1])
    if q_kv_same_len == 1:
        kv_len = q_len
        torch_mask = kt.dec_self_attn_mask(kv_len) * -1e8
        # [1, kv_len, kv_len]
        torch_mask = torch_mask.unsqueeze(0)
        torch_mask = torch_mask.to(torch.float32)
        mask_future = True
    else:
        kv_len = random.randint(1, max_seq_len)
        torch_mask = None
        mask_future = False

    print(
        "(q_len, kv_len, max_batch_tokens, head_dim, batch_head): "
        f"({q_len}, {kv_len}, {max_batch_tokens}, {head_dim}, {batch_head})"
    )
    query = kt.rand((batch_head, q_len, head_dim))
    key = kt.rand((batch_head, max_seq_len, head_dim))
    value = kt.rand((batch_head, max_seq_len, head_dim))
    mask = kt.rand((batch_size, kv_len))
    res = kt.rand((batch_head, q_len, head_dim))
    if kt.dtype == torch.float:
        func = layer_module.torch_sdpa_layer_fp32
    else:
        func = layer_module.torch_sdpa_layer_fp16

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    def custom():
        func(
            query,
            key,
            value,
            mask,
            res,
            max_batch_tokens,
            max_seq_len,
            head_dim,
            nhead,
            0.0,
            batch_size,
            q_len,
            kv_len,
            max_seq_len,
            mask_future,
        )
        return kt.norm_res_list(res)

    def baseline():
        q = query.to(torch.float32)
        k = key[:, :kv_len, :].to(torch.float32)
        v = value[:, :kv_len, :].to(torch.float32)
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / (head_dim**0.5)
        if torch_mask is not None:
            attn_weights += torch_mask
        attn_weights = functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(kt.dtype).to(torch.float32)
        out = torch.matmul(attn_weights, v)
        return kt.norm_res_list(out)

    return custom, baseline


if __name__ == "__main__":
    kt.init(device="cuda:0", nhead=16)
    kt.run(
        [
            "test_sdpa_layer",
        ]
    )
