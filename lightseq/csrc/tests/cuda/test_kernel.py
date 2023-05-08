import sys
from __init__ import lightseq_dir

sys.path.insert(0, lightseq_dir)

import random
import torch
from torch._C import ScriptModule, dtype
from torch.nn import functional

from csrc.pytorch.builder import CudaKernelBuilder
from csrc.tests.util import TestDecorator

cuda_module = CudaKernelBuilder().load()
kt = TestDecorator()


@kt.case()
def test_launch_bias_add_transform_20314():
    batch_size, seq_len = kt.bs_sl()
    hidden_dim = kt.hidden_dim
    nhead = kt.nhead
    head_dim = int(hidden_dim / nhead)
    count = random.randint(1, 20)
    print(
        "(batch_size, seq_len, count, nhead, head_dim): "
        f"({batch_size}, {seq_len}, {count}, {nhead}, {head_dim})"
    )

    qkv = kt.rand((batch_size, seq_len, count, hidden_dim))
    bias = kt.zeros((1, 1, count, hidden_dim))
    custom_res = kt.rand((count, batch_size, nhead, seq_len, head_dim))

    if kt.dtype == torch.float:
        func = cuda_module.torch_launch_bias_add_transform_20314_fp32
    else:
        func = cuda_module.torch_launch_bias_add_transform_20314_fp16

    def custom():
        func(custom_res, qkv, bias, batch_size, seq_len, count, nhead, head_dim)
        return [
            custom_res,
        ]

    def baseline():
        # [batch_size, seq_len, count, hidden_dim]
        base = qkv + bias
        # [count, batch_size, seq_len, hidden_dim]
        base = base.transpose(1, 2).transpose(0, 1)
        base = base.reshape((count, batch_size, seq_len, nhead, head_dim)).transpose(
            2, 3
        )
        return [
            base.contiguous(),
        ]

    return custom, baseline


@kt.case()
def test_launch_bias_add_transform_20314_new():
    batch_size, seq_len = kt.bs_sl()
    hidden_dim = kt.hidden_dim
    nhead = kt.nhead
    head_dim = int(hidden_dim / nhead)
    count = random.randint(1, 3)
    print(
        "(batch_size, seq_len, count, nhead, head_dim): "
        f"({batch_size}, {seq_len}, {count}, {nhead}, {head_dim})"
    )

    qkv = kt.rand((batch_size, seq_len, count, hidden_dim))
    bias = kt.zeros((1, 1, count, hidden_dim))
    custom_q = kt.rand((batch_size, nhead, seq_len, head_dim))
    custom_k = kt.rand((batch_size, nhead, seq_len, head_dim))
    custom_v = kt.rand((batch_size, nhead, seq_len, head_dim))
    base_res = kt.rand((count, batch_size, nhead, seq_len, head_dim))

    if kt.dtype == torch.float:
        cust_func = cuda_module.torch_launch_bias_add_transform_20314_new_fp32
    else:
        cust_func = cuda_module.torch_launch_bias_add_transform_20314_new_fp16

    if kt.dtype == torch.float:
        base_func = cuda_module.torch_launch_bias_add_transform_20314_fp32
    else:
        base_func = cuda_module.torch_launch_bias_add_transform_20314_fp16

    def custom():
        cust_func(
            custom_q,
            custom_k,
            custom_v,
            qkv,
            bias,
            batch_size,
            seq_len,
            count,
            nhead,
            head_dim,
        )
        if count == 1:
            return [custom_q]
        if count == 2:
            return [torch.cat((custom_q, custom_k), dim=0)]
        return [torch.cat((custom_q, custom_k, custom_v), dim=0)]

    def baseline():
        base_func(base_res, qkv, bias, batch_size, seq_len, count, nhead, head_dim)
        return [
            base_res,
        ]

    return custom, baseline


@kt.case()
def test_launch_transform_0213():
    batch_size, seq_len = kt.bs_sl()
    hidden_dim = kt.hidden_dim
    nhead = kt.nhead
    head_dim = int(hidden_dim / nhead)
    print(
        "(batch_size, seq_len, hidden_dim, nhead): "
        f"({batch_size}, {seq_len}, {hidden_dim}, {nhead})"
    )

    vals = kt.rand((batch_size, seq_len, hidden_dim))
    custom_res = kt.rand((batch_size, nhead, seq_len, head_dim))

    if kt.dtype == torch.float:
        func = cuda_module.torch_launch_transform_0213_fp32
    else:
        func = cuda_module.torch_launch_transform_0213_fp16

    # [sz0, sz1, sz2, sz3] -> [sz0, sz2, sz1, sz3]
    def custom():
        func(vals, custom_res, batch_size, seq_len, nhead, head_dim)
        return kt.norm_res_list(custom_res)

    def baseline():
        base = vals.reshape((batch_size, seq_len, nhead, head_dim)).transpose(1, 2)
        return kt.norm_res_list(base)

    return custom, baseline


@kt.case()
def test_launch_transform4d_0213():
    batch_size, seq_len = kt.bs_sl()
    hidden_dim = kt.hidden_dim
    nhead = kt.nhead
    head_dim = int(hidden_dim / nhead)
    trans_count = random.choice([1, 3])
    print(
        "(batch_size, seq_len, hidden_dim, nhead, trans_count): "
        f"({batch_size}, {seq_len}, {hidden_dim}, {nhead}, {trans_count})"
    )

    vals = kt.rand((trans_count, batch_size, nhead, seq_len, head_dim))
    custom_res = kt.rand((batch_size, seq_len, trans_count, nhead, head_dim))

    if kt.dtype == torch.float:
        func = cuda_module.torch_launch_transform4d_0213_fp32
    else:
        func = cuda_module.torch_launch_transform4d_0213_fp16

    # [trans_count, batch_size, nhead, seq_len, head_dim] ->
    # [batch_size, seq_len, trans_count, nhead, head_dim]

    def custom():
        func(custom_res, vals, batch_size, seq_len, hidden_dim, nhead, trans_count)
        return [
            custom_res,
        ]

    def baseline():
        base = vals.permute(1, 3, 0, 2, 4)
        return [
            base.contiguous(),
        ]

    return custom, baseline


@kt.case(atol=1e-3, rtol=1e-3, ntest=20)
def test_launch_attn_softmax():
    batch_size, from_len = kt.bs_sl()
    is_dec_self_attn = random.choice([True, False])
    if is_dec_self_attn:
        to_len = from_len
        is_dec_self_attn_infer = random.choice([True, False])
    else:
        _, to_len = kt.bs_sl(batch_size)
        is_dec_self_attn_infer = False

    if is_dec_self_attn_infer:
        to_len = from_len
        from_len = 1
        beam_size = random.choice([3, 4, 5])
        batch_size *= beam_size

    nhead = kt.nhead
    print(
        "(batch_size, nhead, from_len, to_len, is_dec_self_attn,"
        f" is_dec_self_attn_infer): ({batch_size}, {nhead}, {from_len}, {to_len},"
        f" {is_dec_self_attn}, {is_dec_self_attn_infer})"
    )

    inp = kt.rand((batch_size, nhead, from_len, to_len))
    if is_dec_self_attn:
        mask = kt.dec_self_attn_mask(to_len) * -1e8
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, to_len, to_len]
    else:
        mask = kt.attn_mask(batch_size, to_len) * -1e8
        mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, to_len]
    inp_list = [inp.clone() for _ in range(8)]
    tt = {"repeat_idx": 0}

    if kt.dtype == torch.float:
        func = cuda_module.torch_launch_attn_softmax_fp32
    else:
        func = cuda_module.torch_launch_attn_softmax_fp16

    def custom():
        inp_dup = inp_list[tt["repeat_idx"]]
        tt["repeat_idx"] += 1
        func(
            inp_dup,
            mask,
            batch_size,
            nhead,
            from_len,
            to_len,
            is_dec_self_attn,
            is_dec_self_attn and (not is_dec_self_attn_infer),
        )
        return [
            inp_dup,
        ]

    def baseline():
        f_inp = inp.clone().to(torch.float32)
        f_mask = mask.to(torch.float32)
        if not is_dec_self_attn_infer:
            res = functional.softmax(f_inp + f_mask, dim=-1, dtype=torch.float32)
        else:
            res = functional.softmax(f_inp, dim=-1, dtype=torch.float32)
        return kt.norm_res_list(res)

    return custom, baseline


@kt.case(atol=1e-3, rtol=1e-3, ntest=2)
def test_launch_attn_softmax_new():
    batch_size, from_len = kt.bs_sl()
    is_dec_self_attn = random.choice([True, False])
    if is_dec_self_attn:
        to_len = from_len
        is_dec_self_attn_infer = random.choice([True, False])
    else:
        _, to_len = kt.bs_sl(batch_size)
        is_dec_self_attn_infer = False

    if is_dec_self_attn_infer:
        to_len = from_len
        from_len = 1
        beam_size = random.choice([3, 4, 5])
        batch_size *= beam_size

    nhead = kt.nhead
    print(
        "(batch_size, nhead, from_len, to_len, is_dec_self_attn,"
        f" is_dec_self_attn_infer): ({batch_size}, {nhead}, {from_len}, {to_len},"
        f" {is_dec_self_attn}, {is_dec_self_attn_infer})"
    )

    inp = kt.rand((batch_size, nhead, from_len, to_len))
    if is_dec_self_attn:
        mask = kt.dec_self_attn_mask(to_len) * -1e8
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, to_len, to_len]
    else:
        mask = kt.attn_mask(batch_size, to_len) * -1e8
        mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, to_len]
    inp_list = [inp.clone() for _ in range(8)]
    tt = {"repeat_idx": 0}

    if kt.dtype == torch.float:
        func = cuda_module.torch_launch_attn_softmax_new_fp32
    else:
        func = cuda_module.torch_launch_attn_softmax_new_fp16

    if kt.dtype == torch.float:
        func2 = cuda_module.torch_launch_attn_softmax_fp32
    else:
        func2 = cuda_module.torch_launch_attn_softmax_fp16

    cust_out = torch.empty_like(inp)

    def custom():
        cus_inp = inp_list[tt["repeat_idx"]].clone()
        func(
            cust_out,
            cus_inp,
            mask,
            batch_size,
            nhead,
            from_len,
            to_len,
            is_dec_self_attn,
            is_dec_self_attn and (not is_dec_self_attn_infer),
        )
        return [
            cust_out,
        ]

    def baseline():
        base_inp = inp_list[tt["repeat_idx"]].clone()
        func2(
            base_inp,
            mask,
            batch_size,
            nhead,
            from_len,
            to_len,
            is_dec_self_attn,
            is_dec_self_attn and (not is_dec_self_attn_infer),
        )
        return [
            base_inp,
        ]

    return custom, baseline


@kt.case(atol=1e-2, rtol=1e-3)
def test_launch_attn_softmax_bw():
    nhead = kt.nhead
    batch_size, from_len = kt.bs_sl()
    _, to_len = kt.bs_sl(batch_size)
    print(
        "(batch_size, nhead, from_len, to_len): "
        f"({batch_size}, {nhead}, {from_len}, {to_len})"
    )

    out_grad = kt.rand((batch_size, nhead, from_len, to_len))
    soft_inp = kt.rand((batch_size, nhead, from_len, to_len))

    if kt.dtype == torch.float:
        func = cuda_module.torch_launch_attn_softmax_bw_fp32
    else:
        func = cuda_module.torch_launch_attn_softmax_bw_fp16

    def custom():
        out_grad_dup = out_grad.clone()
        func(out_grad_dup, soft_inp, batch_size * nhead * from_len, to_len)
        return [
            out_grad_dup,
        ]

    def baseline():
        f_soft_inp = soft_inp.to(dtype=torch.float)
        f_out_grad = out_grad.clone().to(dtype=torch.float)
        tsum = f_out_grad * f_soft_inp
        # [b, nh, s, 1]
        tsum = tsum.sum(dim=-1, keepdim=True)
        res = f_soft_inp * (f_out_grad - tsum)
        return kt.norm_res_list(res)

    return custom, baseline


@kt.case(atol=1e-2, rtol=1e-3)
def test_launch_attn_softmax_bw_new():
    nhead = kt.nhead
    batch_size, from_len = kt.bs_sl()
    _, to_len = kt.bs_sl(batch_size)
    print(
        "(batch_size, nhead, from_len, to_len): "
        f"({batch_size}, {nhead}, {from_len}, {to_len})"
    )

    out_grad = kt.rand((batch_size, nhead, from_len, to_len))
    soft_inp = kt.rand((batch_size, nhead, from_len, to_len))
    inp_grad = kt.rand((batch_size, nhead, from_len, to_len))

    if kt.dtype == torch.float:
        base_func = cuda_module.torch_launch_attn_softmax_bw_fp32
    else:
        base_func = cuda_module.torch_launch_attn_softmax_bw_fp16

    if kt.dtype == torch.float:
        cust_func = cuda_module.torch_launch_attn_softmax_bw_new_fp32
    else:
        cust_func = cuda_module.torch_launch_attn_softmax_bw_new_fp16

    inp_grad_dup = inp_grad.clone()

    def custom():
        out_grad_dup = out_grad.clone()
        cust_func(
            inp_grad_dup, out_grad_dup, soft_inp, batch_size * nhead * from_len, to_len
        )
        return [
            inp_grad_dup,
        ]

    def baseline():
        base_out_grad = out_grad.clone()
        base_func(base_out_grad, soft_inp, batch_size * nhead * from_len, to_len)
        return [
            base_out_grad,
        ]

    return custom, baseline


@kt.case()
def test_launch_fused_add2():
    batch_size, seq_len = kt.bs_sl()
    hidden_dim = kt.hidden_dim
    print(f"(batch_size, seq_len, hidden_dim): ({batch_size}, {seq_len}, {hidden_dim})")

    val1 = kt.rand((batch_size, seq_len, hidden_dim))
    val2 = kt.rand((batch_size, seq_len, hidden_dim))
    custom_res = kt.rand((batch_size, seq_len, hidden_dim))

    if kt.dtype == torch.float:
        func = cuda_module.torch_launch_fused_add2_fp32
    else:
        func = cuda_module.torch_launch_fused_add2_fp16

    # [batch_size, seq_len, hidden_dim] ->
    # [batch_size, seq_len, hidden_dim]

    def custom():
        func(custom_res, val1, val2, batch_size, seq_len, hidden_dim)
        return [
            custom_res,
        ]

    def baseline():
        base = val1 + val2
        return [
            base.contiguous(),
        ]

    return custom, baseline


@kt.case(atol=1e-2, rtol=1e-3)
def test_launch_layer_norm():
    batch_size, seq_len = kt.bs_sl()
    bsz_seq = batch_size * seq_len
    hidden_dim = kt.hidden_dim
    with_mean = random.choice([True, False])
    print(
        "(batch_token_num, hidden_dim, with_mean): "
        f"({bsz_seq}, {hidden_dim}, {with_mean})"
    )

    custom_res = kt.rand((bsz_seq, hidden_dim))
    inp = kt.rand((bsz_seq, hidden_dim))
    gamma = kt.rand((hidden_dim))
    beta = kt.rand((hidden_dim))
    vars = kt.rand((bsz_seq))
    means = kt.rand((bsz_seq))

    if kt.dtype == torch.float:
        func = cuda_module.torch_launch_layer_norm_fp32
    else:
        func = cuda_module.torch_launch_layer_norm_fp16

    def custom():
        func(custom_res, vars, means, inp, gamma, beta, bsz_seq, hidden_dim, with_mean)
        return [custom_res, vars, means] if with_mean else [custom_res, vars]

    def baseline():
        base = torch.nn.functional.layer_norm(
            inp, [hidden_dim], gamma, beta, kt.epsilon
        )
        if with_mean:
            return [
                base.contiguous(),
                inp.var(dim=1).contiguous(),
                inp.mean(dim=1).contiguous(),
            ]
        else:
            return [base.contiguous(), inp.var(dim=1).contiguous()]

    return custom, baseline


@kt.case(atol=1e-2, rtol=1e-3)
def test_launch_layer_norm_i8O():
    batch_size, seq_len = kt.bs_sl()
    bsz_seq = batch_size * seq_len
    hidden_dim = kt.hidden_dim
    with_mean = random.choice([True, False])
    print(
        "(batch_token_num, hidden_dim, with_mean): "
        f"({bsz_seq}, {hidden_dim}, {with_mean})"
    )

    # shared weights
    inp = kt.rand((bsz_seq, hidden_dim))
    gamma = kt.rand((hidden_dim))
    beta = kt.rand((hidden_dim))
    vars = kt.rand((bsz_seq))
    means = kt.rand((bsz_seq))
    cmax = kt.topk(inp)

    # custom weights
    custom_res = kt.randint8((bsz_seq, hidden_dim))
    custom_cmask = kt.randuint8((bsz_seq, hidden_dim))

    if kt.dtype == torch.float:
        func = cuda_module.torch_launch_layer_norm_i8_fp32
    else:
        func = cuda_module.torch_launch_layer_norm_i8_fp16

    def custom():
        custom_cmask.zero_()
        test_res = inp.clone()
        func(
            custom_res,
            custom_cmask,
            vars,
            means,
            inp,
            gamma,
            beta,
            cmax,
            bsz_seq,
            hidden_dim,
            with_mean,
        )

        return (
            [custom_res, vars, means, custom_cmask]
            if with_mean
            else [custom_res, vars, custom_cmask]
        )

    def baseline():
        base_res = torch.nn.functional.layer_norm(
            inp, [hidden_dim], gamma, beta, kt.epsilon
        )
        base_res, base_cmask = kt.quantize(base_res, cmax)

        if with_mean:
            return [
                base_res.contiguous(),
                inp.var(dim=1).contiguous(),
                inp.mean(dim=1).contiguous(),
                base_cmask.contiguous(),
            ]
        else:
            return [
                base_res.contiguous(),
                inp.var(dim=1).contiguous(),
                base_cmask.contiguous(),
            ]

    return custom, baseline


@kt.case(atol=1e-3, rtol=1e-2)
def test_launch_ln_bw():
    batch_size, seq_len = kt.bs_sl()
    bsz_seq = batch_size * seq_len
    hidden_dim = kt.hidden_dim
    with_mean = random.choice([True, False])
    fuse_add = random.choice([True, False])
    print(
        "(batch_token_num, hidden_dim, with_mean, fuse_add): "
        f"({bsz_seq}, {hidden_dim}, {with_mean}, {fuse_add})"
    )

    ln_input = kt.rand((bsz_seq, hidden_dim))
    out_grad = kt.rand((bsz_seq, hidden_dim))
    residual_grad = kt.rand((bsz_seq, hidden_dim))
    gamma = kt.rand((hidden_dim))
    betta = kt.rand((hidden_dim))
    gamma_grad = kt.rand((hidden_dim))
    betta_grad = kt.rand((hidden_dim))
    inp_grad = kt.rand((bsz_seq, hidden_dim))

    ln_output = functional.layer_norm(ln_input, [hidden_dim], gamma, betta, kt.epsilon)
    vars = ln_input.var(dim=1).contiguous()
    means = ln_input.mean(dim=1).contiguous()

    if kt.dtype == torch.float:
        func = cuda_module.torch_launch_ln_bw_fp32
    else:
        func = cuda_module.torch_launch_ln_bw_fp16

    inp_or_out = ln_input if with_mean else ln_output

    def custom():
        func(
            gamma_grad,
            betta_grad,
            inp_grad,
            out_grad,
            residual_grad,
            inp_or_out,
            gamma,
            betta,
            vars,
            means,
            bsz_seq,
            hidden_dim,
            with_mean,
            fuse_add,
        )
        # return [inp_grad]
        return [gamma_grad, betta_grad, inp_grad]

    def baseline():
        if with_mean:
            (
                f_out_grad,
                f_input,
                f_vars,
                f_means,
                f_betta,
                f_gamma,
            ) = kt.cast_fp32_tensor([out_grad, ln_input, vars, means, betta, gamma])
            xhat = (f_input - f_means.unsqueeze(1)) * f_vars.rsqrt().unsqueeze(1)
        else:
            f_out_grad, f_out, f_vars, f_betta, f_gamma = kt.cast_fp32_tensor(
                [out_grad, ln_output, vars, betta, gamma]
            )
            xhat = (f_out - f_betta) / f_gamma
        dxhat = f_out_grad * f_gamma
        f_betta_grad = f_out_grad.sum(dim=0)
        f_gamma_grad = (f_out_grad * xhat).sum(dim=0)
        dinp = dxhat.sum(dim=1).unsqueeze(1) + xhat * (dxhat * xhat).sum(
            dim=1
        ).unsqueeze(1)
        dinp = dxhat - dinp / hidden_dim
        dinp = dinp * f_vars.rsqrt().unsqueeze(1)
        if fuse_add:
            dinp = dinp + residual_grad
        return kt.norm_res_list(f_gamma_grad, f_betta_grad, dinp)

    return custom, baseline


@kt.case(atol=1e-3, rtol=1e-2)
def test_launch_ln_i8O_bw():
    batch_size, seq_len = kt.bs_sl()
    bsz_seq = batch_size * seq_len
    hidden_dim = kt.hidden_dim
    with_mean = random.choice([True, False])
    fuse_add = random.choice([True, False])
    print(
        "(batch_token_num, hidden_dim, with_mean, fuse_add): "
        f"({bsz_seq}, {hidden_dim}, {with_mean}, {fuse_add})"
    )

    # shared weights
    inp = kt.rand((bsz_seq, hidden_dim))
    out_grad = kt.rand((bsz_seq, hidden_dim))
    residual_grad = kt.rand((bsz_seq, hidden_dim))
    gamma = kt.rand((hidden_dim))
    betta = kt.rand((hidden_dim))
    # forward
    ln_output = functional.layer_norm(inp, [hidden_dim], gamma, betta, kt.epsilon)
    vars = inp.var(dim=1).contiguous()
    means = inp.mean(dim=1).contiguous()
    cmax = kt.topk(inp)
    cmask = kt.get_cmask(ln_output, cmax)
    inp_or_out = inp if with_mean else ln_output

    # custom weights
    gamma_grad = kt.rand((hidden_dim))
    betta_grad = kt.rand((hidden_dim))
    inp_grad = kt.rand((bsz_seq, hidden_dim))
    cmax_grad = kt.zeros(1)

    if kt.dtype == torch.float:
        func = cuda_module.torch_launch_ln_bw_i8_fp32
    else:
        func = cuda_module.torch_launch_ln_bw_i8_fp16

    def custom():
        func(
            gamma_grad,
            betta_grad,
            inp_grad,
            cmax_grad,
            out_grad,
            residual_grad,
            inp_or_out,
            gamma,
            betta,
            vars,
            means,
            cmask,
            bsz_seq,
            hidden_dim,
            with_mean,
            fuse_add,
        )
        return [gamma_grad, betta_grad, inp_grad, cmax_grad]

    def baseline():
        if with_mean:
            (
                f_out_grad,
                f_input,
                f_vars,
                f_means,
                f_betta,
                f_gamma,
            ) = kt.cast_fp32_tensor([out_grad, inp, vars, means, betta, gamma])
            xhat = (f_input - f_means.unsqueeze(1)) * f_vars.rsqrt().unsqueeze(1)
        else:
            f_out_grad, f_out, f_vars, f_betta, f_gamma = kt.cast_fp32_tensor(
                [out_grad, ln_output, vars, betta, gamma]
            )
            xhat = (f_out - f_betta) / f_gamma
        f_out_grad_inrange = kt.tensor_inrange(f_out_grad, ln_output, cmax)
        f_out_grad_outrange = kt.tensor_outrange(f_out_grad, ln_output, cmax)
        f_cmax_grad = f_out_grad_outrange.sum()
        dxhat = f_out_grad_inrange * f_gamma
        f_betta_grad = f_out_grad_inrange.sum(dim=0)
        f_gamma_grad = (f_out_grad_inrange * xhat).sum(dim=0)
        dinp = dxhat.sum(dim=1).unsqueeze(1) + xhat * (dxhat * xhat).sum(
            dim=1
        ).unsqueeze(1)
        dinp = dxhat - dinp / hidden_dim
        dinp = dinp * f_vars.rsqrt().unsqueeze(1)
        if fuse_add:
            dinp = dinp + residual_grad
        return kt.norm_res_list(f_gamma_grad, f_betta_grad, dinp, f_cmax_grad)

    return custom, baseline


@kt.case(atol=1e-3, rtol=1e-2)
def test_launch_ffn_bias_bwd():
    batch_size, seq_len = kt.bs_sl()
    hidden_dim = kt.hidden_dim
    coef = random.randint(1, 4)
    print(f"(rows, cols): ({batch_size*seq_len}, {coef*hidden_dim})")

    val = kt.rand((batch_size * seq_len, coef * hidden_dim))
    custom_res = kt.rand((coef * hidden_dim))

    if kt.dtype == torch.float:
        func = cuda_module.torch_launch_ffn_bias_bwd_fp32
    else:
        func = cuda_module.torch_launch_ffn_bias_bwd_fp16

    # [batch_size*seq_len, coef*hidden_dim] ->
    # [batch_size*seq_len, coef*hidden_dim]

    def custom():
        func(val, custom_res, batch_size * seq_len, coef * hidden_dim)
        return [
            custom_res,
        ]

    def baseline():
        temp = val.to(torch.float)
        base = torch.sum(temp, 0)
        base = base.to(val)
        return [
            base.contiguous(),
        ]

    return custom, baseline


@kt.case()
def test_launch_concat3_dim1():
    batch_size, seq_len = kt.bs_sl()
    hidden_dim = kt.hidden_dim
    nhead = kt.nhead
    head_dim = int(hidden_dim / nhead)
    assert seq_len > 1
    sl1 = random.randint(1, seq_len - 1)
    sl2 = seq_len - sl1
    beam_size = random.randint(1, 8)
    print(
        f"(batch_size, beam_size, nhead, sl1, sl2, head_dim): {batch_size},"
        f" {beam_size}, {nhead}, {sl1}, {sl2}, {head_dim}"
    )

    inp1 = kt.rand((batch_size, beam_size, nhead, sl1, head_dim))
    inp2 = kt.rand((batch_size, beam_size, nhead, sl2, head_dim))
    custom_res = kt.rand((batch_size, beam_size, nhead, seq_len, head_dim))

    if kt.dtype == torch.float:
        func = cuda_module.torch_launch_concat3_dim1_fp32
    else:
        func = cuda_module.torch_launch_concat3_dim1_fp16

    def custom():
        func(inp1, inp2, custom_res, batch_size * beam_size * nhead, head_dim, sl1, sl2)
        return kt.norm_res_list(custom_res)

    def baseline():
        res = torch.cat((inp1, inp2), dim=3)
        return kt.norm_res_list(res)

    return custom, baseline


@kt.case(dtypes=[torch.float, torch.half], ntest=5, atol=1e-2, rtol=1e-2)
def test_launch_dropout_relu_bias():
    batch_size, seq_len = kt.bs_sl()
    hidden_dim = kt.hidden_dim
    print("test shape:", (batch_size, seq_len, hidden_dim))

    test_input = kt.rand((batch_size, seq_len, hidden_dim))
    test_bias = kt.rand((hidden_dim,))
    test_out_cus = kt.rand((batch_size, seq_len, hidden_dim))
    test_mask_cus = torch.rand((batch_size, seq_len, hidden_dim)).to(
        dtype=torch.uint8, device="cuda:0"
    )

    if kt.dtype == torch.float:
        cus_func = cuda_module.torch_launch_ls_dropout_relu_bias_fp32
    else:
        cus_func = cuda_module.torch_launch_ls_dropout_relu_bias_fp16

    def custom():
        cus_func(
            test_out_cus,
            test_mask_cus,
            test_input,
            test_bias,
            batch_size * seq_len,
            hidden_dim,
            0,
        )

        return test_out_cus

    def baseline():
        test_out_base = torch.nn.functional.relu(test_input + test_bias)
        test_out_base = torch.nn.functional.dropout(test_out_base, p=0)

        return test_out_base

    return custom, baseline


@kt.case(dtypes=[torch.float, torch.half], ntest=5, atol=1e-2, rtol=1e-2)
def test_launch_dropout_gelu_bias():
    batch_size, seq_len = kt.bs_sl()
    hidden_dim = kt.hidden_dim
    print("test shape:", (batch_size, seq_len, hidden_dim))

    test_input = kt.rand((batch_size, seq_len, hidden_dim))
    test_bias = kt.rand((hidden_dim,))
    test_out_cus = kt.rand((batch_size, seq_len, hidden_dim))
    test_mask_cus = torch.rand((batch_size, seq_len, hidden_dim)).to(
        dtype=torch.uint8, device="cuda:0"
    )

    if kt.dtype == torch.float:
        cus_func = cuda_module.torch_launch_ls_dropout_gelu_bias_fp32
    else:
        cus_func = cuda_module.torch_launch_ls_dropout_gelu_bias_fp16

    def custom():
        cus_func(
            test_out_cus,
            test_mask_cus,
            test_input,
            test_bias,
            batch_size * seq_len,
            hidden_dim,
            0,
        )

        return test_out_cus

    def baseline():
        test_out_base = torch.nn.functional.gelu(test_input + test_bias)
        test_out_base = torch.nn.functional.dropout(test_out_base, p=0)

        return test_out_base

    return custom, baseline


@kt.case(dtypes=[torch.float, torch.half], ntest=5, atol=1e-2, rtol=1e-2)
def test_launch_dropout_relu_bias_bwd():
    batch_size, seq_len = kt.bs_sl()
    hidden_dim = kt.hidden_dim * 4
    print("test shape:", (batch_size, seq_len, hidden_dim))

    test_input = kt.rand((batch_size, seq_len, hidden_dim))
    test_bias = kt.rand((hidden_dim))
    test_out_grad = kt.rand((batch_size, seq_len, hidden_dim))
    test_in_grad_cus = kt.rand((batch_size, seq_len, hidden_dim))
    test_bias_grad_cus = kt.rand((hidden_dim))
    test_mask = torch.ones((batch_size, seq_len, hidden_dim)).to(
        dtype=torch.uint8,
        device="cuda:0",
    )

    if kt.dtype == torch.float:
        cus_func = cuda_module.torch_launch_ls_dropout_relu_bias_bwd_fp32
    else:
        cus_func = cuda_module.torch_launch_ls_dropout_relu_bias_bwd_fp16

    def custom():
        cus_func(
            test_in_grad_cus,
            test_bias_grad_cus,
            test_mask,
            test_input,
            test_bias,
            test_out_grad,
            batch_size * seq_len,
            hidden_dim,
            0.1,
        )

        return test_in_grad_cus, test_bias_grad_cus

    def baseline():
        temp = test_out_grad.float() * test_mask * (1 / (1 - 0.1))
        test_in_grad_base = temp * ((test_input.float() + test_bias.float()) > 0)
        test_bias_grad_base = torch.sum(test_in_grad_base, (0, 1), dtype=torch.float)

        return test_in_grad_base.to(kt.dtype), test_bias_grad_base.to(kt.dtype)

    return custom, baseline


@kt.case(dtypes=[torch.float, torch.half], ntest=5, atol=1e-2, rtol=1e-2)
def test_launch_dropout_gelu_bias_bwd():
    batch_size, seq_len = kt.bs_sl()
    hidden_dim = kt.hidden_dim * 4
    print("test shape:", (batch_size, seq_len, hidden_dim))

    test_input = kt.rand((batch_size, seq_len, hidden_dim))
    test_input.requires_grad_()
    test_bias = kt.rand((hidden_dim))
    test_bias.requires_grad_()
    test_out_grad = kt.rand((batch_size, seq_len, hidden_dim))
    test_in_grad_cus = kt.rand((batch_size, seq_len, hidden_dim))
    test_bias_grad_cus = kt.rand((hidden_dim))
    test_mask = torch.ones((batch_size, seq_len, hidden_dim)).to(
        dtype=torch.uint8,
        device="cuda:0",
    )

    if kt.dtype == torch.float:
        cus_func = cuda_module.torch_launch_ls_dropout_gelu_bias_bwd_fp32
    else:
        cus_func = cuda_module.torch_launch_ls_dropout_gelu_bias_bwd_fp16

    temp = torch.nn.functional.gelu(test_input + test_bias)
    base_out = torch.nn.functional.dropout(temp, p=0)
    base_out = base_out * test_out_grad
    base_out = base_out.sum()

    def custom():
        cus_func(
            test_in_grad_cus,
            test_bias_grad_cus,
            test_mask,
            test_input,
            test_bias,
            test_out_grad,
            batch_size * seq_len,
            hidden_dim,
            0,
        )

        return test_in_grad_cus, test_bias_grad_cus

    def baseline():
        if test_input.grad is not None:
            test_input.grad.zero_()
            test_bias.grad.zero_()
        base_out.backward(retain_graph=True)

        return [
            test_input.grad.contiguous().detach(),
            test_bias.grad.contiguous().detach(),
        ]

    return custom, baseline


from torch_crf import CRF


@kt.case(dtypes=[torch.half], atol=5.0)
def test_crf():
    batch_size = 129
    seq_len = 33
    num_tags = 41
    torch_mask = ~kt.attn_mask(batch_size, seq_len, torch.bool)
    # torch_mask = kt.ones((batch_size, seq_len)).to(torch.bool)
    ls_mask = torch_mask.clone()
    ls_mask = (~ls_mask).to(dtype=torch.uint8)

    emissions = kt.rand((batch_size, seq_len, num_tags))
    crf = CRF(num_tags, batch_first=True)
    crf.to(kt.device, torch.float)
    crf.start_transitions.data.to(torch.half).to(torch.float)
    crf.end_transitions.data.to(torch.half).to(torch.float)
    crf.transitions.data.to(torch.half).to(torch.float)

    """
    torch_launch_viterbi(const torch::Tensor &start_transition,
                          const torch::Tensor &end_transition,
                          const torch::Tensor &transition,
                          const torch::Tensor &emission,
                          const torch::Tensor &mask, torch::Tensor &score,
                          torch::Tensor &next_score, torch::Tensor &history,
                          torch::Tensor &best_tags, int num_tags, int seq_len,
                          int batch_size)
    """
    start_transition = (
        crf.start_transitions.data.clone().detach().to(kt.dtype).contiguous()
    )
    end_transition = crf.end_transitions.data.clone().detach().to(kt.dtype).contiguous()
    transitions = (
        crf.transitions.data.clone().detach().transpose(0, 1).to(kt.dtype).contiguous()
    )

    best_score = kt.zeros((batch_size)).to(dtype=torch.float)
    history = kt.ones((batch_size, seq_len, num_tags)).to(dtype=torch.int32)
    best_tags = kt.zeros((batch_size, seq_len)).to(dtype=torch.int32)

    cus_func = cuda_module.torch_launch_viterbi_fp16

    def custom():
        cus_func(
            start_transition,
            end_transition,
            transitions,
            emissions,
            ls_mask,
            best_score,
            history,
            best_tags,
            num_tags,
            seq_len,
            batch_size,
        )
        return [best_tags, best_score]

    def baseline():
        res, best_score = crf.decode(emissions, torch_mask, pad_tag=-1)
        return [
            res.detach().to(torch.int32),
            best_score.detach().to(torch.float),
        ]

    return custom, baseline


@kt.case(atol=4, rtol=1e-2)
def test_launch_dropout_relu_bias_i8I_i8O():
    batch_size, seq_len = kt.bs_sl()
    hidden_dim = kt.hidden_dim
    print("test shape:", (batch_size, seq_len, hidden_dim))

    # shared weights
    inp = kt.randint8((batch_size, seq_len, hidden_dim))
    bias = kt.rand((hidden_dim,))
    mask = kt.ones((batch_size, seq_len, hidden_dim)).to(torch.uint8)
    cmax_out = (kt.topk(inp) / 127).to(kt.dtype)
    cmax_in = (kt.topk(inp) / 127).to(kt.dtype)

    # custom weights
    custom_res = kt.randint8((batch_size, seq_len, hidden_dim))
    custom_cmask_out = kt.randuint8((batch_size, seq_len, hidden_dim))

    if kt.dtype == torch.float:
        cus_func = cuda_module.torch_launch_ls_quant_dropout_relu_bias_fp32
    else:
        cus_func = cuda_module.torch_launch_ls_quant_dropout_relu_bias_fp16

    def custom():
        custom_cmask_in = kt.ones((batch_size, seq_len, hidden_dim)).to(torch.uint8)
        cus_func(
            custom_res,
            custom_cmask_out,
            custom_cmask_in,
            custom_cmask_in,
            inp,
            bias,
            cmax_out,
            cmax_in,
            batch_size * seq_len,
            hidden_dim,
            0,
        )

        return [custom_res, custom_cmask_in]

    def baseline():
        inp_dq = kt.dequantize(inp, cmax_out)
        out_base = torch.nn.functional.relu(inp_dq + bias)
        out_base = torch.nn.functional.dropout(out_base, p=0)
        out_base, cmask_in_base = kt.quantize(out_base, cmax_in)
        cmask_in_base |= mask

        return [out_base, cmask_in_base]

    return [custom, baseline]


@kt.case(atol=4, rtol=1e-2)
def test_launch_dropout_gelu_bias_i8I_i8O():
    batch_size, seq_len = kt.bs_sl()
    hidden_dim = kt.hidden_dim
    print("test shape:", (batch_size, seq_len, hidden_dim))

    # shared weights
    inp = kt.randint8((batch_size, seq_len, hidden_dim))
    bias = kt.rand((hidden_dim,))
    mask = kt.ones((batch_size, seq_len, hidden_dim)).to(torch.uint8)
    cmax_out = (kt.topk(inp) / 127).to(kt.dtype)
    cmax_in = (kt.topk(inp) / 127).to(kt.dtype)

    # custom weights
    custom_res = kt.randint8((batch_size, seq_len, hidden_dim))
    custom_cmask_out = kt.randuint8((batch_size, seq_len, hidden_dim))

    if kt.dtype == torch.float:
        cus_func = cuda_module.torch_launch_ls_quant_dropout_gelu_bias_fp32
    else:
        cus_func = cuda_module.torch_launch_ls_quant_dropout_gelu_bias_fp16

    def custom():
        custom_cmask_in = kt.ones((batch_size, seq_len, hidden_dim)).to(torch.uint8)
        cus_func(
            custom_res,
            custom_cmask_out,
            custom_cmask_in,
            custom_cmask_in,
            inp,
            bias,
            cmax_out,
            cmax_in,
            batch_size * seq_len,
            hidden_dim,
            0,
        )

        return [custom_res, custom_cmask_in]

    def baseline():
        inp_dq = kt.dequantize(inp, cmax_out)
        out_base = torch.nn.functional.gelu(inp_dq + bias)
        out_base = torch.nn.functional.dropout(out_base, p=0)
        out_base, cmask_in_base = kt.quantize(out_base, cmax_in)
        cmask_in_base |= mask

        return [out_base, cmask_in_base]

    return [custom, baseline]


@kt.case(atol=1e-2, rtol=1e-2)
def test_launch_dropout_relu_bias_i8I_i8O_bwd():
    batch_size, seq_len = kt.bs_sl()
    hidden_dim = kt.hidden_dim * 4
    print("test shape:", (batch_size, seq_len, hidden_dim))

    # shared weights
    inp = kt.randint8((batch_size, seq_len, hidden_dim))
    bias = kt.rand((hidden_dim,))
    out_grad = kt.rand((batch_size, seq_len, hidden_dim))
    mask = kt.ones((batch_size, seq_len, hidden_dim)).to(torch.uint8)
    cmax_out = (kt.topk(inp) / 127).to(kt.dtype)
    cmax_in = (kt.topk(inp) / 127).to(kt.dtype)

    inp_dq = kt.dequantize(inp, cmax_out, float_out=True)
    res = torch.nn.functional.relu(inp_dq + bias)
    res = torch.nn.functional.dropout(res, p=0)

    # custom weights
    custom_inp_grad = kt.rand((batch_size, seq_len, hidden_dim))
    custom_bias_grad = kt.rand((hidden_dim))
    custom_cmask_out = kt.get_cmask(inp, cmax_out)
    custom_cmask_in = kt.get_cmask(res, cmax_in)
    custom_cmax_in_grad = kt.zeros(1)
    custom_cmax_out_grad = kt.zeros(1)

    if kt.dtype == torch.float:
        cus_func = cuda_module.torch_launch_ls_quant_dropout_relu_bias_bwd_fp32
    else:
        cus_func = cuda_module.torch_launch_ls_quant_dropout_relu_bias_bwd_fp16

    def custom():
        cus_func(
            custom_inp_grad,
            custom_bias_grad,
            custom_cmax_out_grad,
            custom_cmax_in_grad,
            mask,
            inp,
            cmax_out,
            custom_cmask_out,
            custom_cmask_in,
            bias,
            out_grad,
            batch_size * seq_len,
            hidden_dim,
            0,
        )
        return custom_inp_grad, custom_bias_grad, custom_cmax_in_grad

    def baseline():
        out_grad_inrange = kt.tensor_inrange(out_grad, res, cmax_in)
        out_grad_outrange = kt.tensor_outrange(out_grad, res, cmax_in)
        base_cmax_in_grad = out_grad_outrange.sum()

        temp = out_grad_inrange * mask
        base_inp_grad = temp * ((inp_dq + bias) > 0)
        base_bias_grad = torch.sum(base_inp_grad, (0, 1))
        return base_inp_grad, base_bias_grad, base_cmax_in_grad

    return custom, baseline


@kt.case(atol=1e-2, rtol=1e-2)
def test_launch_dropout_gelu_bias_i8I_i8O_bwd():
    batch_size, seq_len = kt.bs_sl()
    hidden_dim = kt.hidden_dim * 4
    print("test shape:", (batch_size, seq_len, hidden_dim))

    # shared weights
    inp = kt.randint8((batch_size, seq_len, hidden_dim))
    bias = kt.rand((hidden_dim,))
    out_grad = kt.rand((batch_size, seq_len, hidden_dim))
    mask = kt.ones((batch_size, seq_len, hidden_dim)).to(torch.uint8)
    cmax_out = (kt.topk(inp) / 127).to(kt.dtype)
    cmax_in = (kt.topk(inp) / 127).to(kt.dtype)
    inp_dq = kt.dequantize(inp, cmax_out)
    gelu_inp = inp_dq + bias
    gelu_inp.requires_grad_()
    gelu_out = torch.nn.functional.gelu(gelu_inp)
    gelu_out_sum = gelu_out.sum()
    res = torch.nn.functional.dropout(gelu_out, p=0)

    # custom weights
    custom_inp_grad = kt.rand((batch_size, seq_len, hidden_dim))
    custom_bias_grad = kt.rand((hidden_dim))
    custom_cmask_out = kt.get_cmask(inp, cmax_out)
    custom_cmask_in = kt.get_cmask(res, cmax_in)
    custom_cmax_in_grad = kt.zeros(1)
    custom_cmax_out_grad = kt.zeros(1)

    if kt.dtype == torch.float:
        cus_func = cuda_module.torch_launch_ls_quant_dropout_gelu_bias_bwd_fp32
    else:
        cus_func = cuda_module.torch_launch_ls_quant_dropout_gelu_bias_bwd_fp16

    def custom():
        cus_func(
            custom_inp_grad,
            custom_bias_grad,
            custom_cmax_out_grad,
            custom_cmax_in_grad,
            mask,
            inp,
            cmax_out,
            custom_cmask_out,
            custom_cmask_in,
            bias,
            out_grad,
            batch_size * seq_len,
            hidden_dim,
            0,
        )

        return custom_inp_grad, custom_bias_grad, custom_cmax_in_grad

    def baseline():
        out_grad_inrange = kt.tensor_inrange(out_grad, res, cmax_in)
        out_grad_outrange = kt.tensor_outrange(out_grad, res, cmax_in)
        base_cmax_in_grad = out_grad_outrange.sum()

        temp = out_grad_inrange * mask
        if gelu_inp.grad is not None:
            gelu_inp.grad.zero_()
        gelu_out_sum.backward(retain_graph=True)
        base_inp_grad = temp * gelu_inp.grad
        base_bias_grad = torch.sum(base_inp_grad, (0, 1))

        return base_inp_grad, base_bias_grad, base_cmax_in_grad

    return custom, baseline


@kt.case()
def test_launch_quant_bias_dropout_residual():
    batch_size, seq_len = kt.bs_sl()
    hidden_dim = kt.hidden_dim
    print("test shape:", (batch_size, seq_len, hidden_dim))

    # shared weights
    inp = kt.randint8((batch_size, seq_len, hidden_dim))
    residual = kt.rand((batch_size, seq_len, hidden_dim)) * 0.1
    bias = kt.rand((hidden_dim,)) * 0.1
    mask = kt.ones((batch_size, seq_len, hidden_dim)).to(torch.uint8)
    # cmax_out = (kt.topk(inp) / 127).to(kt.dtype)
    cmax = torch.tensor([16]).to(kt.dtype).to(kt.device)

    # custom weights
    custom_res = kt.rand((batch_size, seq_len, hidden_dim))

    if kt.dtype == torch.float:
        cus_func = cuda_module.torch_launch_ls_quant_bias_dropout_residual_fp32
    else:
        cus_func = cuda_module.torch_launch_ls_quant_bias_dropout_residual_fp16

    def custom():
        cus_func(
            custom_res,
            mask,
            inp,
            cmax,
            bias,
            residual,
            batch_size * seq_len,
            hidden_dim,
            0,
        )

        return [custom_res]

    def baseline():
        inp_dq = kt.dequantize(inp, cmax)
        out_base = inp_dq + bias
        out_base = torch.nn.functional.dropout(out_base, p=0)
        out_base = out_base + residual

        return [out_base]

    return [custom, baseline]


@kt.case()
def test_launch_quant_bias_add_transform_20314():
    batch_size, seq_len = kt.bs_sl()
    hidden_dim = kt.hidden_dim
    nhead = kt.nhead
    head_dim = int(hidden_dim / nhead)
    count = random.randint(1, 20)
    print(
        "(batch_size, seq_len, count, nhead, head_dim): "
        f"({batch_size}, {seq_len}, {count}, {nhead}, {head_dim})"
    )

    # shared weights
    qkv = kt.randint8((batch_size, seq_len, count, hidden_dim))
    bias = kt.zeros((1, 1, count, hidden_dim))
    cmax = (kt.topk(qkv) / 127).to(kt.dtype)
    cmask = kt.randuint8((batch_size, seq_len, count, hidden_dim))

    # custom weights
    custom_res = kt.rand((count, batch_size, nhead, seq_len, head_dim))

    if kt.dtype == torch.float:
        func = cuda_module.torch_launch_quant_bias_add_transform_20314_fp32
    else:
        func = cuda_module.torch_launch_quant_bias_add_transform_20314_fp16

    def custom():
        func(
            custom_res,
            cmask,
            qkv,
            bias,
            cmax,
            batch_size,
            seq_len,
            count,
            nhead,
            head_dim,
        )
        return [
            custom_res,
        ]

    def baseline():
        # [batch_size, seq_len, count, hidden_dim]
        qkv_dq = kt.dequantize(qkv, cmax)
        base = qkv_dq + bias
        # [count, batch_size, seq_len, hidden_dim]
        base = base.transpose(1, 2).transpose(0, 1)
        base = base.reshape((count, batch_size, seq_len, nhead, head_dim)).transpose(
            2, 3
        )
        return [
            base.contiguous(),
        ]

    return custom, baseline


@kt.case()
def test_launch_quant_transform4d_0213():
    batch_size, seq_len = kt.bs_sl()
    hidden_dim = kt.hidden_dim
    nhead = kt.nhead
    head_dim = int(hidden_dim / nhead)
    trans_count = random.choice([1, 3])
    print(
        "(batch_size, seq_len, hidden_dim, nhead, trans_count): "
        f"({batch_size}, {seq_len}, {hidden_dim}, {nhead}, {trans_count})"
    )

    # shared weights
    vals = kt.rand((trans_count, batch_size, nhead, seq_len, head_dim))
    cmax = kt.topk(vals).to(kt.dtype)

    # custom weights
    custom_cmask = kt.randuint8((batch_size, seq_len, trans_count, nhead, head_dim))
    custom_res = kt.randint8((batch_size, seq_len, trans_count, nhead, head_dim))

    if kt.dtype == torch.float:
        func = cuda_module.torch_launch_quant_transform4d_0213_fp32
    else:
        func = cuda_module.torch_launch_quant_transform4d_0213_fp16

    # [trans_count, batch_size, nhead, seq_len, head_dim] ->
    # [batch_size, seq_len, trans_count, nhead, head_dim]

    def custom():
        custom_cmask.zero_()
        func(
            custom_res,
            custom_cmask,
            vals,
            cmax,
            batch_size,
            seq_len,
            hidden_dim,
            nhead,
            trans_count,
        )
        return [custom_res, custom_cmask]

    def baseline():
        base = vals.permute(1, 3, 0, 2, 4)
        base_q, base_cmask = kt.quantize(base, cmax)
        return [base_q.contiguous(), base_cmask.contiguous()]

    return custom, baseline


@kt.case()
def test_torch_launch_ls_quantize():
    batch_size, seq_len = kt.bs_sl()
    hidden_dim = kt.hidden_dim

    print(
        "(batch_size, seq_len, hidden_dim): " f"({batch_size}, {seq_len}, {hidden_dim})"
    )

    # shared weights
    inputs = kt.rand((batch_size, seq_len, hidden_dim))
    base_cmax = torch.tensor(16, dtype=kt.dtype, device=kt.device)
    custom_cmax = torch.tensor([16, 16], dtype=kt.dtype, device=kt.device)
    cmask = kt.randuint8((batch_size, seq_len, hidden_dim))
    igemm_alpha = torch.tensor(1.0, dtype=torch.float, device=kt.device)

    # custom weights
    custom_res = kt.randint8((batch_size, seq_len, hidden_dim))

    if kt.dtype == torch.float:
        func = cuda_module.torch_launch_ls_quantize_fp32
    else:
        func = cuda_module.torch_launch_ls_quantize_fp16

    def custom():
        cmask.zero_()
        func(custom_res, cmask, igemm_alpha, inputs, custom_cmax, inputs.numel())
        return [custom_res, cmask]

    def baseline():
        base, base_mask = kt.quantize(inputs, base_cmax)
        base_mask *= 4

        return [base.contiguous(), base_mask.contiguous()]

    return custom, baseline


@kt.case()
def test_torch_launch_ls_dequantize():
    batch_size, seq_len = kt.bs_sl()
    hidden_dim = kt.hidden_dim

    print(
        "(batch_size, seq_len, hidden_dim): " f"({batch_size}, {seq_len}, {hidden_dim})"
    )

    # shared weights
    inputs = kt.randint8((batch_size, seq_len, hidden_dim))
    base_cmax = torch.tensor(16, dtype=kt.dtype, device=kt.device)
    custom_cmax = torch.tensor([16, 16], dtype=kt.dtype, device=kt.device)

    # custom weights
    custom_res = kt.rand((batch_size, seq_len, hidden_dim)).to(kt.dtype)

    if kt.dtype == torch.float:
        func = cuda_module.torch_launch_ls_dequantize_fp32
    else:
        func = cuda_module.torch_launch_ls_dequantize_fp16

    def custom():
        func(custom_res, inputs, custom_cmax, inputs.numel())
        return [custom_res]

    def baseline():
        base = kt.dequantize(inputs, base_cmax, True)
        base = base.to(kt.dtype)

        return [base.contiguous()]

    return custom, baseline


@kt.case(atol=1e-2, rtol=1e-3)
def test_torch_launch_fake_quantize():
    batch_size, seq_len = kt.bs_sl()
    hidden_dim = kt.hidden_dim

    print(
        "(batch_size, seq_len, hidden_dim): " f"({batch_size}, {seq_len}, {hidden_dim})"
    )

    # shared weights
    inputs = kt.rand((batch_size, seq_len, hidden_dim))
    base_cmax = torch.tensor(16, dtype=kt.dtype, device=kt.device)
    custom_cmax = torch.tensor([16, 16], dtype=kt.dtype, device=kt.device)
    cmask = kt.randuint8((batch_size, seq_len, hidden_dim))
    igemm_alpha = torch.tensor(1.0, dtype=torch.float, device=kt.device)

    custom_res = kt.rand((batch_size, seq_len, hidden_dim))

    if kt.dtype == torch.float:
        func = cuda_module.torch_launch_fake_quantize_fp32
    else:
        func = cuda_module.torch_launch_fake_quantize_fp16

    def custom():
        cmask.zero_()
        func(
            cmask,
            igemm_alpha,
            custom_res,
            inputs,
            custom_cmax,
            inputs.numel(),
        )
        return [custom_res, cmask]

    def baseline():
        base, base_mask = kt.quantize(inputs, base_cmax)
        base = kt.dequantize(base, base_cmax)

        return [base.contiguous(), base_mask.contiguous()]

    return custom, baseline


@kt.case(atol=1e-2, rtol=1e-3, dtypes=[torch.float, torch.half])
def test_split_rotary_position_qkv():
    batch_size, offset_seq_len = kt.bs_sl()
    nhead = kt.nhead
    head_dim = 128
    seq_len = 1
    seq_len = random.randint(1, 2048)
    offset_seq_len = random.randint(0, 2048 - seq_len)
    outshape = kt.rand((batch_size, nhead, seq_len, head_dim))

    cachek = kt.rand((batch_size, nhead, offset_seq_len, head_dim))
    cachev = kt.rand((batch_size, nhead, offset_seq_len, head_dim))
    q_tensor = kt.rand((batch_size, seq_len, nhead, head_dim))
    k_tensor = kt.rand((batch_size, seq_len, nhead, head_dim))
    v_tensor = kt.rand((batch_size, seq_len, nhead, head_dim))
    qkv_tensor = torch.cat((q_tensor, k_tensor, v_tensor), dim=2)

    out_cachek = torch.cat((cachek, outshape), dim=2)
    out_cachev = torch.cat((cachev, outshape), dim=2)

    func = None
    if kt.dtype == torch.float:
        func = cuda_module.torch_launch_split_rotary_position_fp32
    elif kt.dtype == torch.half:
        func = cuda_module.torch_launch_split_rotary_position_fp16

    custom_q = torch.empty_like(q_tensor)

    def custom():
        func(
            qkv_tensor,
            custom_q,
            out_cachek,
            out_cachev,
            batch_size,
            nhead,
            offset_seq_len,
            seq_len,
            head_dim,
        )
        return [custom_q.contiguous(), out_cachek.contiguous(), out_cachev.contiguous()]

    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2).float().to(device="cuda:0") / head_dim)
    )
    t = torch.arange(2048, device="cuda:0", dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_cached = emb.cos()[None, None, :, :].to(device="cuda:0", dtype=kt.dtype)
    sin_cached = emb.sin()[None, None, :, :].to(device="cuda:0", dtype=kt.dtype)

    def baseline():
        trans_q = q_tensor.transpose(1, 2)
        trans_k = k_tensor.transpose(1, 2)
        trans_v = v_tensor.transpose(1, 2)
        kv_seq_len = offset_seq_len + seq_len
        cos = cos_cached[:, :, :kv_seq_len, ...]
        sin = sin_cached[:, :, :kv_seq_len, ...]
        gather_indices = (
            (torch.arange(seq_len) + offset_seq_len)[None, None, :, None]
            .repeat(batch_size, cos.shape[1], 1, cos.shape[3])
            .to("cuda:0")
        )
        cos = torch.gather(
            cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices
        )
        sin = torch.gather(
            sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices
        )

        def rotate_half(x):
            """Rotates half the hidden dims of the input."""
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_out = (trans_q * cos) + (rotate_half(trans_q) * sin)
        k_out = (trans_k * cos) + (rotate_half(trans_k) * sin)
        k_out = torch.cat((cachek, k_out), dim=2)
        v_out = torch.cat((cachev, trans_v), dim=2)
        return [q_out.contiguous(), k_out.contiguous(), v_out.contiguous()]

    return custom, baseline


from transformers import LlamaModel
from transformers.activations import SiLUActivation


@kt.case(atol=1e-3, rtol=1e-4, dtypes=[torch.float, torch.half])
def test_silu_elewise_product():
    batch_size, seq_len = 1, 256
    hidden_size = 13824
    inpA = kt.rand((batch_size, seq_len, hidden_size))
    inpB = kt.rand((batch_size, seq_len, hidden_size))
    custom_outC = torch.empty_like(inpA)

    act_func = SiLUActivation()
    func = (
        cuda_module.torch_silu_elewise_product_fp32
        if kt.dtype == torch.float
        else cuda_module.torch_silu_elewise_product_fp16
    )

    def custom():
        func(inpA, inpB, custom_outC, batch_size, seq_len, hidden_size)
        return [custom_outC.contiguous()]

    def baseline():
        output = act_func(inpA) * inpB
        return [output.contiguous()]

    return custom, baseline


@kt.case(atol=1e-3, rtol=1e-4, dtypes=[torch.float, torch.half])
def test_rms_layer_norm():  # torch_rms_layer_norm
    batch_size, seq_len = 1, 1  # kt.bs_sl()
    hidden_size = 5120
    inp = kt.rand((batch_size, seq_len, hidden_size))
    scale = kt.rand((hidden_size))
    custom_out = torch.empty_like(inp)
    rms_out = kt.rand((batch_size, seq_len))

    func = (
        cuda_module.torch_rms_layer_norm_fp32
        if kt.dtype == torch.float
        else cuda_module.torch_rms_layer_norm_fp16
    )

    def custom():
        func(inp, scale, custom_out, rms_out, batch_size * seq_len, hidden_size, 1e-6)
        return [rms_out.contiguous(), custom_out.contiguous()]

    def baseline():
        # output = act_func(inpA) * inpB
        variance = inp.to(torch.float32).pow(2).mean(-1, keepdim=True)
        rms_var = torch.rsqrt(variance + 1e-6).to(dtype=kt.dtype)
        hidden_states = inp * rms_var
        output = (scale * hidden_states).to(dtype=kt.dtype)
        return [rms_var.contiguous(), output.contiguous()]

    return custom, baseline


if __name__ == "__main__":
    kt.init(device="cuda:0", nhead=16)
    kt.run(
        [
            # "test_rms_layer_norm",
            # "test_silu_elewise_product",
            "test_split_rotary_position_qkv",
            # "test_launch_transform_0213",
            # "test_launch_bias_add_transform_20314",
            # "test_launch_transform4d_0213",
            # "test_launch_bias_add_transform_20314_new",
            # "test_launch_fused_add2",
            # "test_launch_ffn_bias_bwd",
            # # "test_launch_attn_softmax", # need to fix
            # "test_launch_attn_softmax_new",
            # # "test_launch_attn_softmax_bw", # need to fix
            # "test_launch_attn_softmax_bw_new",
            # "test_launch_layer_norm",
            # # "test_launch_ln_bw", # need to fix
            # "test_launch_concat3_dim1",
            # # "test_adam", # need to fix
            # "test_launch_dropout_relu_bias",
            # "test_launch_dropout_relu_bias_bwd",
            # "test_launch_dropout_gelu_bias",
            # # "test_launch_dropout_gelu_bias_bwd", # need to fix
            # # "test_launch_layer_norm_i8O", # need to fix
            # # "test_launch_ln_i8O_bw", # need to fix
            # "test_launch_dropout_relu_bias_i8I_i8O",
            # # "test_launch_dropout_relu_bias_i8I_i8O_bwd", # need to fix
            # "test_launch_dropout_gelu_bias_i8I_i8O",
            # # "test_launch_dropout_gelu_bias_i8I_i8O_bwd", # need to fix
            # "test_launch_quant_bias_dropout_residual",
            # "test_launch_quant_bias_add_transform_20314",
            # # "test_launch_quant_transform4d_0213", # need to fix
            # # "test_torch_launch_ls_quantize", # need to fix
            # "test_torch_launch_ls_dequantize",
            # # "test_torch_launch_fake_quantize", # need to fix
            # # "test_crf", # need to fix
        ]
    )
