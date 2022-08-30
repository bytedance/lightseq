from itertools import zip_longest
import math

import torch
from torch import nn
from torch.autograd import Function

from lightseq.training.ops.pytorch.layer_base import TransformerEncoderLayerBase
from lightseq.training.ops.pytorch.builder import TransformerBuilder
from lightseq.training.ops.pytorch.quantization import (
    weight_quant_config,
    act_quant_config,
    relu_quant_config,
)
from lightseq.training.ops.pytorch.util import (
    copy_para,
    state_dict,
    calc_offset,
)

transformer_cuda_module = TransformerBuilder().load()


_all_layer_grads = dict()


class LSTransformerEncoderFunc(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        input_mask,
        parameters,
        config,
    ):
        cuda_module = transformer_cuda_module
        forward_func = (
            cuda_module.transformer_encoder_layer_fw_fp16
            if config.fp16
            else cuda_module.transformer_encoder_layer_fw_fp32
        )
        if config.fp16:
            input = input.to(torch.half)
            input_mask = input_mask.to(torch.half)

        (output,) = forward_func(
            config.layer_id,
            input,
            input_mask,
            config.training,
            config.pre_layer_norm,
            config.quant_mode,
        )

        if config.is_grad_enabled and config.training:
            ctx.save_for_backward(output, input, input_mask)
            ctx.config = config
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert ctx.config.training

        cuda_module = transformer_cuda_module
        backward_func = (
            cuda_module.transformer_encoder_layer_bw_fp16
            if ctx.config.fp16
            else cuda_module.transformer_encoder_layer_bw_fp32
        )

        output, input, input_mask = ctx.saved_tensors
        if ctx.config.fp16:
            grad_output = grad_output.to(torch.half)
            output = output.to(torch.half)
            input = input.to(torch.half)
            input_mask = input_mask.to(torch.half)
        (grad_input,) = backward_func(
            ctx.config.layer_id, grad_output, output, input, input_mask
        )

        grad = _all_layer_grads[ctx.config.layer_id]

        return (grad_input, None, grad, None)


class LSTransformerEncoderLayer(TransformerEncoderLayerBase):
    """Initialize the Lightseq Transformer Encoder Layer.

    Static variable:
        layer_id: The layer-index counter starting from 0 and incrementing by 1 every time a layer object is instantiated,
        e.g. if a model has 24 transformer layers, layer_id goes from 0 to 23.
    Arguments:
        config: An object of LSTransformerEncoderLayer config, see get_config

        initial_weights: Optional: Only used for unit test

        initial_biases: Optional: Only used for unit test
    """

    layer_id = 0

    def __init__(self, config, initial_weights=None, initial_biases=None):
        super(LSTransformerEncoderLayer, self).__init__()

        self.config = config
        self.config.layer_id = LSTransformerEncoderLayer.layer_id
        LSTransformerEncoderLayer.layer_id = LSTransformerEncoderLayer.layer_id + 1

        print("Lightseq Transformer config is ", self.config.__dict__)

        self.quant_mode = False

        if self.config.local_rank >= 0:
            torch.cuda.set_device(self.config.local_rank)

        self.create_cpp_layer()

        hs = self.config.hidden_size
        ims = self.config.intermediate_size
        self.hs = hs
        self.ims = ims
        self.para_offset = LSTransformerEncoderLayer.gen_offset(hs, ims)
        self.para = nn.Parameter(torch.Tensor(self.para_offset[-1]))

        if initial_weights is None or initial_biases is None:
            self.init_transformer_weights()
            return

        # For testing only.
        qkv_w = [ele.detach().clone() for ele in initial_weights[:3]]
        qkv_w = torch.cat(qkv_w, dim=0)
        weights = [qkv_w] + [copy_para(ele) for ele in initial_weights[3:]]

        qkv_b = [ele.detach().clone() for ele in initial_biases[:3]]
        qkv_b = torch.cat(qkv_b, dim=0)
        biases = [qkv_b] + [copy_para(ele) for ele in initial_biases[3:]]

        idx = 0
        for w, b in zip_longest(weights, biases):
            if w is not None:
                cur_para = self._get_weights(idx)
                assert cur_para.numel() == w.numel()
                cur_para.copy_(w.view(-1))
                idx += 1

            if b is not None:
                cur_para = self._get_weights(idx)
                assert cur_para.numel() == b.numel()
                cur_para.copy_(b.view(-1))
                idx += 1

    @staticmethod
    def gen_offset(hidden_size, intermediate_size):
        hs, ims = hidden_size, intermediate_size
        sizes = [
            hs * hs * 3,  # attn_qkvw
            hs * 3,  # attn_qkvb
            hs * hs,  # attn_ow
            hs,  # attn_ob
            hs,  # attn_nw
            hs,  # attn_nb
            hs * ims,  # inter_w
            ims,  # inter_b
            hs * ims,  # output_w
            hs,  # output_b
            hs,  # ffn_nw
            hs,  # ffn_nb
            12,  # clip_max
        ]
        offsets = calc_offset(sizes)
        return offsets

    def create_cpp_layer(self):

        # create the layer in cuda kernels.
        cuda_module = transformer_cuda_module
        create_layer_func = (
            cuda_module.create_transformer_encoder_layer_fp16
            if self.config.fp16
            else cuda_module.create_transformer_encoder_layer_fp32
        )

        create_layer_func(
            self.config.layer_id,
            self.config.max_batch_tokens,
            self.config.max_seq_len,
            self.config.hidden_size,
            self.config.nhead,
            self.config.intermediate_size,
            self.config.attn_prob_dropout_ratio,
            self.config.activation_dropout_ratio,
            self.config.hidden_dropout_ratio,
            self.config.pre_layer_norm,
            self.config.activation_fn,
            False,  # mask_future_tokens
        )

    def _get_weights(self, i):
        return self.para.data.narrow(
            0, self.para_offset[i], self.para_offset[i + 1] - self.para_offset[i]
        )

    def calc_bound(self, w):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1.0 / math.sqrt(fan_in)
        return bound

    def init_transformer_weights(self):
        hs = self.config.hidden_size
        ims = self.config.intermediate_size
        attn_qkvw = self._get_weights(0).view(-1, hs)
        nn.init.xavier_uniform_(attn_qkvw, 1.0 / math.sqrt(2.0))
        bound = self.calc_bound(attn_qkvw)
        nn.init.uniform_(self._get_weights(1), -bound, bound)

        nn.init.xavier_uniform_(self._get_weights(2).view(-1, hs), 1.0)
        nn.init.zeros_(self._get_weights(3))

        nn.init.ones_(self._get_weights(4))
        nn.init.zeros_(self._get_weights(5))

        inter_w = self._get_weights(6).view(ims, hs)
        nn.init.kaiming_uniform_(inter_w, math.sqrt(5.0))
        bound = self.calc_bound(inter_w)
        nn.init.uniform_(self._get_weights(7), -bound, bound)

        output_w = self._get_weights(8).view(hs, ims)
        nn.init.kaiming_uniform_(output_w, math.sqrt(5.0))
        bound = self.calc_bound(output_w)
        nn.init.uniform_(self._get_weights(9), -bound, bound)

        nn.init.ones_(self._get_weights(10))
        nn.init.zeros_(self._get_weights(11))

        act_cmax = act_quant_config.amax.tolist()
        wei_cmax = weight_quant_config.amax.tolist()
        init_clip_max = torch.tensor([act_cmax, wei_cmax, act_cmax] * 4)
        self._get_weights(12).copy_(init_clip_max)

    def params_dict(self):
        """
        Returns:
            weight: dict
            bias: dict
        """

        def copy_and_view(m, shape=None):
            if shape is None:
                shape = (-1,)
            return m.data.clone().view(*shape)

        self_attn_qkvw = self._get_weights(0)
        self_attn_qw, self_attn_kw, self_attn_vw = self_attn_qkvw.split(
            self.hs * self.hs, 0
        )
        self_attn_qkvb = self._get_weights(1)
        self_attn_qb, self_attn_kb, self_attn_vb = self_attn_qkvb.split(self.hs, 0)

        weight = {
            "self_attn.q_proj": copy_and_view(self_attn_qw, (self.hs, self.hs)),
            "self_attn.k_proj": copy_and_view(self_attn_kw, (self.hs, self.hs)),
            "self_attn.v_proj": copy_and_view(self_attn_vw, (self.hs, self.hs)),
            "self_attn.out_proj": copy_and_view(
                self._get_weights(2), (self.hs, self.hs)
            ),
            "self_attn_layer_norm": copy_and_view(self._get_weights(4), (self.hs,)),
            "fc1": copy_and_view(self._get_weights(6), (self.ims, self.hs)),
            "fc2": copy_and_view(self._get_weights(8), (self.hs, self.ims)),
            "final_layer_norm": copy_and_view(self._get_weights(10), (self.hs,)),
            "clip_max": copy_and_view(self._get_weights(12), (12,)),
        }
        bias = {
            "self_attn.q_proj": copy_and_view(self_attn_qb),
            "self_attn.k_proj": copy_and_view(self_attn_kb),
            "self_attn.v_proj": copy_and_view(self_attn_vb),
            "self_attn.out_proj": copy_and_view(self._get_weights(3)),
            "self_attn_layer_norm": copy_and_view(self._get_weights(5)),
            "fc1": copy_and_view(self._get_weights(7)),
            "fc2": copy_and_view(self._get_weights(9)),
            "final_layer_norm": copy_and_view(self._get_weights(11)),
        }
        return weight, bias

    def __assign_layer_weight_grad(self):
        param = (
            self.para_16
            if self.config.fp16 and self.para.dtype != torch.half
            else self.para
        )
        if self.config.layer_id in _all_layer_grads:
            return
        global transformer_cuda_module
        cuda_module = transformer_cuda_module
        if self.config.fp16:
            func = cuda_module.assign_layer_weight_grad_fp16
        else:
            func = cuda_module.assign_layer_weight_grad_fp32
        grad = torch.zeros_like(param)
        func(param, grad, "TransformerEncoderLayer", self.config.layer_id)
        _all_layer_grads[self.config.layer_id] = grad

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        destination = state_dict(
            self, destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        return destination

    def forward(self, hidden_states, encoder_padding_mask, **kwargs):
        # encoder_padding_mask is a mask for the input sequence
        # sizes are [batch_size, seq_len] or [seq_len] when batch_size = 1
        # masked value should be 1.0, unmasked value should be 0.0

        self.config.training = self.training
        self.config.is_grad_enabled = torch.is_grad_enabled()
        self.config.quant_mode = self.quant_mode

        hidden_states = hidden_states.contiguous()
        encoder_padding_mask = (
            (encoder_padding_mask * -1e8).type_as(hidden_states).contiguous()
        )
        if self.config.fp16 and self.para.dtype != torch.half:
            if hasattr(self, "para_16"):
                self.para_16.copy_(self.para.to(torch.half))
            else:
                self.register_buffer("para_16", self.para.clone().detach().half())

        self.__assign_layer_weight_grad()
        bs, sl, dim = hidden_states.size()
        if bs * sl > self.config.max_batch_tokens:
            raise ValueError(
                f"Batch token numbers {bs * sl} exceeds the limit"
                f" {self.config.max_batch_tokens}."
            )
        if sl > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {sl} exceeds the limit {self.config.max_seq_len}."
            )
        if len(encoder_padding_mask.size()) == 1:
            assert bs == 1 and sl == encoder_padding_mask.size(0)
        else:
            assert bs == encoder_padding_mask.size(
                0
            ) and sl == encoder_padding_mask.size(1)
        output = LSTransformerEncoderFunc.apply(
            hidden_states,
            encoder_padding_mask,
            self.para,
            self.config,
        )

        return output.to(self.para)

    def disable_quant(self):
        self.quant_mode = False

    def enable_quant(self):
        self.quant_mode = True
