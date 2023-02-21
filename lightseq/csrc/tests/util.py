import random
import time
from collections import OrderedDict

import numpy as np
import torch

max_batch_tokens = 9216
max_seq_len = 1024


class TestDecorator(object):
    def __init__(self):
        self.all_case = OrderedDict()
        self.dtypes = [torch.float, torch.half]
        self.dtype = None
        self.max_batch_tokens = max_batch_tokens
        self.max_seq_len = max_seq_len
        self.epsilon = 1e-8

    def init(self, device, nhead):
        # device: str. e.g. "cuda:0"
        self.device = torch.device(device)
        self._device_str = device
        assert nhead % 4 == 0
        self.nhead = nhead

    def bs_sl(self, batch_size=None, enable_quant=False):
        if batch_size is None:
            seq_len = random.randint(1, self.max_seq_len)
            max_batch_size = self.max_batch_tokens // seq_len
            batch_size = random.randint(1, max_batch_size)
        else:
            max_seq_len = min(self.max_batch_tokens // batch_size, self.max_seq_len)
            seq_len = random.randint(1, max_seq_len)

        if enable_quant and seq_len < 8:
            return self.bs_sl(batch_size, enable_quant)

        return batch_size, seq_len

    @property
    def hidden_dim(self):
        upbound = 1024 // self.nhead
        head_dim = random.choice(range(1, upbound + 1))
        hs = head_dim * self.nhead * self.io_factor
        return hs

    @property
    def io_factor(self):
        if self.dtype == torch.float32:
            return 4
        else:
            return 8

    def cast_fp32_tensor(self, tlist):
        return [ele.to(torch.float32) for ele in tlist]

    def move(self, data):
        return data.to(self.device, dtype=self.dtype)

    def norm_res_list(self, rlist):
        return [ele.to(dtype=self.dtype).contiguous() for ele in rlist]

    def get_cmask(self, x, cmax):
        x_cmask = (x <= -cmax).to(dtype=torch.uint8) * 4 + (x >= cmax).to(
            dtype=torch.uint8
        ) * 2
        return x_cmask

    def quantize(self, x, cmax):
        x, cmax = x.float(), cmax.float()
        qmask = self.get_cmask(x, cmax)
        dequant_scale = cmax / 127
        x = x / dequant_scale
        x = (x + 0.5).floor()
        x = x.clamp(-127, 127).to(dtype=torch.int8)
        return x, qmask

    def dequantize(self, x, cmax, float_out=False):
        x = x.float()
        cmax = cmax.float()
        dequant_scale = cmax / 127
        x = x * dequant_scale
        x = x.clamp(-cmax, cmax)
        if not float_out:
            x = x.to(self.dtype)
        return x

    def topk(self, x, k=100):
        return x.abs().flatten().topk(k)[0][-1]

    def tensor_inrange(self, x, y, cmax):
        x, y, cmax = x.float(), y.float(), cmax.float()
        out = torch.where(y.abs() < cmax, x, self.zeros(1).to(x.dtype))
        return out.to(self.dtype)

    def tensor_outrange(self, x, y, cmax):
        x, y, cmax = x.float(), y.float(), cmax.float()
        out = torch.where(y.abs() >= cmax, x, self.zeros(1).to(x.dtype))
        out = torch.where(y <= -cmax, -out, out)
        return out.to(self.dtype)

    def rand(self, shape):
        return self.move((torch.rand(shape) - 0.5) * 2)

    def randint8(self, shape):
        return torch.randint(-127, 128, shape).to(self.device, dtype=torch.int8)

    def randuint8(self, shape):
        return torch.randint(0, 257, shape).to(self.device, dtype=torch.uint8)

    def randint(self, low, high, shape):
        return torch.randint(low, high, shape).to(self.device, dtype=torch.long)

    def ones(self, shape):
        return self.move(torch.ones(shape))

    def zeros(self, shape):
        return self.move(torch.zeros(shape))

    def attn_mask(self, batch_size, seq_len, dtype=None):
        """
        1 for padding tokens , 0 for non-padding tokens
        """
        if dtype is None:
            dtype = self.dtype
        mask = torch.zeros((batch_size, seq_len))
        for b in range(batch_size):
            valid_seq_len = random.randint(1, seq_len)
            mask[b, valid_seq_len:] = 1
        return mask.to(self.device, dtype=dtype)

    def dec_self_attn_mask(self, seq_len, dtype=None):
        """
        e.g. if seq_len = 3
        return:
        0 1 1
        0 0 1
        0 0 0
        """
        if dtype is None:
            dtype = self.dtype
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask.to(self.device, dtype=dtype)

    def case(self, dtypes=list(), ntest=5, nrepeat=5, rtol=1e-5, atol=1e-5):
        if not dtypes:
            dtypes = self.dtypes

        def decorator(func):
            self.all_case[func.__name__] = (func, dtypes, ntest, nrepeat, rtol, atol)
            return func

        return decorator

    def assert_allclose(self, tlist1, tlist2, rtol, atol):
        """
        tlist1 and tlist2 are list of torch.tensor.
        """
        passed = True
        assert len(tlist1) == len(tlist2)
        for i in range(len(tlist1)):

            def convert_tensor(obj):
                if torch.is_tensor(obj):
                    return obj
                return torch.from_numpy(obj)

            t1 = convert_tensor(tlist1[i])
            t2 = convert_tensor(tlist2[i])

            # fast allclose
            res = torch.allclose(
                t1.flatten(), t2.flatten(), rtol=rtol, atol=atol, equal_nan=False
            )
            if res:
                continue
            passed = False
            print("torch.allclose failed, use numpy.allclose to log detail.")
            t1 = t1.cpu().numpy().flatten()
            t2 = t2.cpu().numpy().flatten()
            try:
                diff_mask = np.isclose(t1, t2, rtol=rtol, atol=atol)
                print("Unmatched x:", t1[~diff_mask])
                print("Unmatched y:", t2[~diff_mask])
                np.testing.assert_allclose(
                    t1, t2, rtol=rtol, atol=atol, verbose=True, equal_nan=False
                )
            except Exception as ex:
                print(f"Unmatches in the {i}-th tensor.")
                print(ex)
                continue
        if not passed:
            exit(0)

    def test(self, custom, baseline, nrepeat, rtol, atol):
        """
        (custom() − baseline()) <= atol + rtol * abs(baseline)
        """

        def core(func):
            res = func()  # warmup for GPU
            self.assert_allclose(res, res, rtol, atol)
            timing = list()
            for i in range(nrepeat):
                if self._device_str != "cpu":
                    torch.cuda.synchronize(device=self.device)
                begin = time.time()
                cur_res = func()
                if self._device_str != "cpu":
                    torch.cuda.synchronize(device=self.device)
                # In seconds
                timing.append(time.time() - begin)
                self.assert_allclose(cur_res, res, rtol, atol)
            return res, np.mean(timing) * 1000

        print("Run baseline...")
        baseline_res, baseline_time = core(baseline)
        print("Run custom...")
        custom_res, custom_time = core(custom)

        print("Compare the results of custom and baseline...")
        self.assert_allclose(custom_res, baseline_res, rtol, atol)
        print(
            "Test passed. Time of custom/baseline (ms): %.3f / %.3f, speedup: %.3f"
            % (custom_time, baseline_time, baseline_time / custom_time)
        )

    def run(self, case_names=None):
        if case_names is None:
            case_names = self.all_case.keys()
        for cn in case_names:
            assert cn in self.all_case, f"name: {cn}, Illegal case name to be tested"
            func, dtypes, ntest, nrepeat, rtol, atol = self.all_case[cn]
            for i in range(ntest):
                for dtype in dtypes:
                    self.dtype = dtype
                    print(
                        f">>>>>>>>>>>>>>>>>>>>>>{cn}, ntest [{i}], dtype [{dtype}], shape {self.bs_sl() + (self.hidden_dim,)}:"
                    )
                    custom, baseline = func()
                    if self._device_str != "cpu":
                        torch.cuda.synchronize(device=self.device)
                    self.test(custom, baseline, nrepeat, rtol, atol)


def flat_dim(idxs, dims):
    assert len(idxs) == len(dims) or len(idxs) == len(dims) + 1
    base = 1
    res = 0
    dims = dims[::-1]
    idxs = idxs[::-1]
    if len(idxs) == len(dims) + 1:
        dims.append(0)
        for idx, dim in zip(idxs, dims):
            assert idx < dim
            res += idx * base
            base *= dim
    return res


def expand_dim(idx, dims):
    res = [0] * len(dims)
    for i, d in enumerate(dims[::-1]):
        res[i] = idx % d
        idx //= d
        if idx == 0:
            break
        assert idx == 0
    return res[::-1]


def get_fairseq_enc_params(fairseq_layer):
    initial_weights = []
    initial_biases = []
    if hasattr(fairseq_layer.self_attn, "qkv_proj"):
        hidden_size = fairseq_layer.self_attn.out_proj.weight.shape[0]
        initial_weights.extend(
            fairseq_layer.self_attn.qkv_proj.weight.detach()
            .clone()
            .split(hidden_size, 0)
        )
        initial_biases.extend(
            fairseq_layer.self_attn.qkv_proj.bias.detach().clone().split(hidden_size, 0)
        )
    else:
        initial_weights.append(fairseq_layer.self_attn.q_proj.weight.detach().clone())
        initial_biases.append(fairseq_layer.self_attn.q_proj.bias.detach().clone())
        initial_weights.append(fairseq_layer.self_attn.k_proj.weight.detach().clone())
        initial_biases.append(fairseq_layer.self_attn.k_proj.bias.detach().clone())
        initial_weights.append(fairseq_layer.self_attn.v_proj.weight.detach().clone())
        initial_biases.append(fairseq_layer.self_attn.v_proj.bias.detach().clone())
    initial_weights.append(fairseq_layer.self_attn.out_proj.weight.detach().clone())
    initial_biases.append(fairseq_layer.self_attn.out_proj.bias.detach().clone())
    initial_weights.append(fairseq_layer.self_attn_layer_norm.weight.detach().clone())
    initial_biases.append(fairseq_layer.self_attn_layer_norm.bias.detach().clone())

    initial_weights.append(fairseq_layer.fc1.weight.detach().clone())
    initial_biases.append(fairseq_layer.fc1.bias.detach().clone())
    initial_weights.append(fairseq_layer.fc2.weight.detach().clone())
    initial_biases.append(fairseq_layer.fc2.bias.detach().clone())
    initial_weights.append(fairseq_layer.final_layer_norm.weight.detach().clone())
    initial_biases.append(fairseq_layer.final_layer_norm.bias.detach().clone())

    clip_max = torch.stack(
        [
            fairseq_layer.self_attn.qkv_proj.input_quant.clip.clip_value_max.detach().clone(),
            fairseq_layer.self_attn.qkv_proj.weight_quant._amax.detach().clone(),
            fairseq_layer.self_attn.qkv_proj.output_quant._amax.detach().clone(),
            fairseq_layer.self_attn.out_proj.input_quant.clip.clip_value_max.detach().clone(),
            fairseq_layer.self_attn.out_proj.weight_quant._amax.detach().clone(),
            fairseq_layer.self_attn.out_proj.output_quant._amax.detach().clone(),
            fairseq_layer.fc1.input_quant.clip.clip_value_max.detach().clone(),
            fairseq_layer.fc1.weight_quant._amax.detach().clone(),
            fairseq_layer.fc1.output_quant._amax.detach().clone(),
            fairseq_layer.fc2.input_quant.clip.clip_value_max.detach().clone(),
            fairseq_layer.fc2.weight_quant._amax.detach().clone(),
            fairseq_layer.fc2.output_quant._amax.detach().clone(),
            # torch.tensor(16).to(
            #     fairseq_layer.self_attn.qkv_proj.input_quant.clip.clip_value_max
            # ),
        ]
    )

    initial_weights.append(clip_max)
    return initial_weights, initial_biases


def get_fairseq_dec_params(fairseq_layer):
    initial_weights = []
    initial_biases = []

    if hasattr(fairseq_layer.self_attn, "qkv_proj"):
        hidden_size = fairseq_layer.self_attn.out_proj.weight.shape[0]
        initial_weights.extend(
            fairseq_layer.self_attn.qkv_proj.weight.detach()
            .clone()
            .split(hidden_size, 0)
        )
        initial_biases.extend(
            fairseq_layer.self_attn.qkv_proj.bias.detach().clone().split(hidden_size, 0)
        )
    else:
        initial_weights.append(fairseq_layer.self_attn.q_proj.weight.detach().clone())
        initial_biases.append(fairseq_layer.self_attn.q_proj.bias.detach().clone())
        initial_weights.append(fairseq_layer.self_attn.k_proj.weight.detach().clone())
        initial_biases.append(fairseq_layer.self_attn.k_proj.bias.detach().clone())
        initial_weights.append(fairseq_layer.self_attn.v_proj.weight.detach().clone())
        initial_biases.append(fairseq_layer.self_attn.v_proj.bias.detach().clone())
    initial_weights.append(fairseq_layer.self_attn.out_proj.weight.detach().clone())
    initial_biases.append(fairseq_layer.self_attn.out_proj.bias.detach().clone())
    initial_weights.append(fairseq_layer.self_attn_layer_norm.weight.detach().clone())
    initial_biases.append(fairseq_layer.self_attn_layer_norm.bias.detach().clone())

    initial_weights.append(fairseq_layer.encoder_attn.q_proj.weight.detach().clone())
    initial_biases.append(fairseq_layer.encoder_attn.q_proj.bias.detach().clone())
    initial_weights.append(fairseq_layer.encoder_attn.k_proj.weight.detach().clone())
    initial_biases.append(fairseq_layer.encoder_attn.k_proj.bias.detach().clone())
    initial_weights.append(fairseq_layer.encoder_attn.v_proj.weight.detach().clone())
    initial_biases.append(fairseq_layer.encoder_attn.v_proj.bias.detach().clone())
    initial_weights.append(fairseq_layer.encoder_attn.out_proj.weight.detach().clone())
    initial_biases.append(fairseq_layer.encoder_attn.out_proj.bias.detach().clone())
    initial_weights.append(
        fairseq_layer.encoder_attn_layer_norm.weight.detach().clone()
    )
    initial_biases.append(fairseq_layer.encoder_attn_layer_norm.bias.detach().clone())

    initial_weights.append(fairseq_layer.fc1.weight.detach().clone())
    initial_biases.append(fairseq_layer.fc1.bias.detach().clone())
    initial_weights.append(fairseq_layer.fc2.weight.detach().clone())
    initial_biases.append(fairseq_layer.fc2.bias.detach().clone())
    initial_weights.append(fairseq_layer.final_layer_norm.weight.detach().clone())
    initial_biases.append(fairseq_layer.final_layer_norm.bias.detach().clone())

    clip_max = torch.stack(
        [
            fairseq_layer.self_attn.qkv_proj.input_quant.clip.clip_value_max.detach().clone(),
            fairseq_layer.self_attn.qkv_proj.weight_quant._amax.detach().clone(),
            fairseq_layer.self_attn.qkv_proj.output_quant._amax.detach().clone(),
            fairseq_layer.self_attn.out_proj.input_quant.clip.clip_value_max.detach().clone(),
            fairseq_layer.self_attn.out_proj.weight_quant._amax.detach().clone(),
            fairseq_layer.self_attn.out_proj.output_quant._amax.detach().clone(),
            fairseq_layer.encoder_attn.q_proj.input_quant.clip.clip_value_max.detach().clone(),
            fairseq_layer.encoder_attn.q_proj.weight_quant._amax.detach().clone(),
            fairseq_layer.encoder_attn.q_proj.output_quant._amax.detach().clone(),
            fairseq_layer.encoder_attn.out_proj.input_quant.clip.clip_value_max.detach().clone(),
            fairseq_layer.encoder_attn.out_proj.weight_quant._amax.detach().clone(),
            fairseq_layer.encoder_attn.out_proj.output_quant._amax.detach().clone(),
            fairseq_layer.fc1.input_quant.clip.clip_value_max.detach().clone(),
            fairseq_layer.fc1.weight_quant._amax.detach().clone(),
            fairseq_layer.fc1.output_quant._amax.detach().clone(),
            fairseq_layer.fc2.input_quant.clip.clip_value_max.detach().clone(),
            fairseq_layer.fc2.weight_quant._amax.detach().clone(),
            fairseq_layer.fc2.output_quant._amax.detach().clone(),
            fairseq_layer.encoder_attn.q_proj.input_quant.clip.clip_value_max.detach().clone(),
            fairseq_layer.encoder_attn.q_proj.weight_quant._amax.detach().clone(),
            fairseq_layer.encoder_attn.q_proj.output_quant._amax.detach().clone(),
            fairseq_layer.encoder_attn.q_proj.input_quant.clip.clip_value_max.detach().clone(),
            fairseq_layer.encoder_attn.q_proj.weight_quant._amax.detach().clone(),
            fairseq_layer.encoder_attn.q_proj.output_quant._amax.detach().clone(),
        ]
    )

    initial_weights.append(clip_max)
    initial_biases.append(None)

    return initial_weights, initial_biases


def split_custom_layer_grad(layer):
    res = []
    for i in range(1, len(layer.para_offset)):
        lidx, ridx = layer.para_offset[i - 1], layer.para_offset[i]
        cur_grad = layer.para.grad.data[lidx:ridx].clone().detach()
        res.append(cur_grad.contiguous())
    return res


def copy_grad_from_paras(para_list):
    res = []
    for para in para_list:
        if para.grad is not None:
            grad = para.grad.data.clone().detach().contiguous()
        else:
            grad = torch.zeros_like(para)
        res.append(grad)
    return res


def copy_cmax_grad_from_paras(para_list):
    res = []
    for para in para_list:
        if para.input_quant.clip.clip_value_max.grad is not None:
            grad = (
                para.input_quant.clip.clip_value_max.grad.data.clone()
                .detach()
                .contiguous()
            )
        else:
            grad = torch.zeros_like(para)
        res.append(grad)

    return [torch.Tensor(res)]
