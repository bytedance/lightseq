import random

import torch
from torch.nn import functional

from lightseq.training.ops.pytorch.builder import KernelBuilder, AdamBuilder
from tests.util import TestDecorator, cast_fp32_tensor

cuda_module = KernelBuilder().load()
adam_module = AdamBuilder().load()
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

    # [batch_size, seq_len, hidden_dim] ->
    # [batch_size, nhead, seq_len, head_dim]

    def custom():
        func(custom_res, vals, batch_size, seq_len, hidden_dim, nhead)
        return [
            custom_res,
        ]

    def baseline():
        base = vals.reshape((batch_size, seq_len, nhead, head_dim)).transpose(1, 2)
        return [
            base.contiguous(),
        ]

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
        return kt.norm_res_list(
            [
                res,
            ]
        )

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
        return kt.norm_res_list(
            [
                res,
            ]
        )

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
        base = torch.nn.functional.layer_norm(inp, [hidden_dim], gamma, beta, 1e-8)
        if with_mean:
            return [
                base.contiguous(),
                inp.var(dim=1).contiguous(),
                inp.mean(dim=1).contiguous(),
            ]
        else:
            return [base.contiguous(), inp.var(dim=1).contiguous()]

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
    epsilon = 1e-12

    ln_input = kt.rand((bsz_seq, hidden_dim))
    out_grad = kt.rand((bsz_seq, hidden_dim))
    residual_grad = kt.rand((bsz_seq, hidden_dim))
    gamma = kt.rand((hidden_dim))
    betta = kt.rand((hidden_dim))
    gamma_grad = kt.rand((hidden_dim))
    betta_grad = kt.rand((hidden_dim))
    inp_grad = kt.rand((bsz_seq, hidden_dim))

    ln_output = functional.layer_norm(ln_input, [hidden_dim], gamma, betta, epsilon)
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
            f_out_grad, f_input, f_vars, f_means, f_betta, f_gamma = cast_fp32_tensor(
                [out_grad, ln_input, vars, means, betta, gamma]
            )
            xhat = (f_input - f_means.unsqueeze(1)) * f_vars.rsqrt().unsqueeze(1)
        else:
            f_out_grad, f_out, f_vars, f_betta, f_gamma = cast_fp32_tensor(
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
        # return kt.norm_res_list([dinp])
        return kt.norm_res_list([f_gamma_grad, f_betta_grad, dinp])

    return custom, baseline


@kt.case()
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
        return kt.norm_res_list([custom_res])

    def baseline():
        res = torch.cat((inp1, inp2), dim=3)
        return kt.norm_res_list([res])

    return custom, baseline


@kt.case(dtypes=[torch.float32])
def test_adam():
    batch_size, seq_len = kt.bs_sl()
    hidden_dim = kt.hidden_dim
    cus_p = kt.rand((batch_size, seq_len, hidden_dim * 32))
    cus_out_p = kt.rand((batch_size, seq_len, hidden_dim * 32))
    cus_exp_avg = kt.rand((batch_size, seq_len, hidden_dim * 32))
    cus_exp_avg_sq = kt.rand((batch_size, seq_len, hidden_dim * 32))
    cus_grad = kt.rand((batch_size, seq_len, hidden_dim * 32))

    base_p = cus_p.detach().clone()
    base_out_p = cus_out_p.detach().clone()
    base_exp_avg = cus_exp_avg.detach().clone()
    base_exp_avg_sq = cus_exp_avg_sq.detach().clone()
    base_grad = cus_grad.detach().clone()

    print("total parameters {}".format(cus_p.numel()))

    def custom():
        adam_module.adam(
            cus_p,
            cus_out_p,
            cus_exp_avg,
            cus_exp_avg_sq,
            cus_grad,
            5e-4,
            0.9,
            0.98,
            1e-8,
            1,
            10,
            1,
            1,
            1e-4,
        )
        return [cus_p]

    def baseline():
        adam_module.apex_adam(
            base_p,
            base_out_p,
            base_exp_avg,
            base_exp_avg_sq,
            base_grad,
            5e-4,
            0.9,
            0.98,
            1e-8,
            1,
            10,
            1,
            1,
            1e-4,
        )
        return [base_p]

    return custom, baseline


@kt.case(dtypes=[torch.float, torch.half], ntest=5, atol=1e-2, rtol=1e-2)
def test_launch_dropout_relu_bias():
    batch_size, seq_len = kt.bs_sl()
    hidden_dim = kt.hidden_dim
    print("test shape:", (batch_size, seq_len, hidden_dim))

    test_input = kt.rand((batch_size, seq_len, hidden_dim))
    test_bias = kt.rand((hidden_dim,))
    test_out_base = kt.rand((batch_size, seq_len, hidden_dim))
    test_out_cus = kt.rand((batch_size, seq_len, hidden_dim))
    test_mask_base = torch.rand((batch_size, seq_len, hidden_dim)).to(
        dtype=torch.uint8,
        device="cuda:0",
    )
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
    test_out_base = kt.rand((batch_size, seq_len, hidden_dim))
    test_out_cus = kt.rand((batch_size, seq_len, hidden_dim))
    temp = kt.rand((batch_size, seq_len, hidden_dim))
    test_mask_base = torch.rand((batch_size, seq_len, hidden_dim)).to(
        dtype=torch.uint8,
        device="cuda:0",
    )
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
        temp = test_out_grad * test_mask * (1 / (1 - 0.1))
        test_in_grad_base = temp * ((test_input + test_bias) > 0)
        test_bias_grad_base = torch.sum(test_in_grad_base, (0, 1))

        return test_in_grad_base, test_bias_grad_base

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


if __name__ == "__main__":
    kt.init(device="cuda:0", nhead=16)
    kernel_list = [
        "test_launch_transform_0213",
        "test_launch_bias_add_transform_20314",
        "test_launch_transform4d_0213",
        "test_launch_fused_add2",
        "test_launch_ffn_bias_bwd",
        "test_launch_attn_softmax",
        "test_launch_attn_softmax_bw",
        "test_launch_layer_norm",
        "test_launch_ln_bw",
        "test_launch_concat3_dim1",
        "test_adam",
        "test_launch_dropout_gelu_bias",
        "test_launch_dropout_relu_bias",
        "test_launch_dropout_relu_bias_bwd",
        "test_launch_dropout_gelu_bias_bwd",
    ]
    kt.run(kernel_list)
