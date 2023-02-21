import sys
from . import lightseq_dir

sys.path.insert(0, lightseq_dir)

import copy
import torch
import math
from dataclasses import dataclass

from training.ops.pytorch.layer_base import TransformerEncoderLayerBase
from training.ops.pytorch.util import (
    copy_para,
    state_dict,
    calc_offset,
)

from csrc.pytorch.builder.cuda_layer_builder import CudaLayerBuilder

cuda_layer_module = CudaLayerBuilder().load()

_all_layer_grads = dict()


class LSTransformerEncoderFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        input_mask,
        config,
    ):
        cuda_module = cuda_layer_module
        forward_func = (
            cuda_module.transformer_encoder_layer_fw_fp16
            if config.fp16
            else cuda_module.transformer_encoder_layer_fw_fp32
        )
        if config.fp16:
            input = input.to(torch.half)
            input_mask = input_mask.to(torch.half)

        (output,) = forward_func(config.layer_id, input, input_mask)

        # if config.is_grad_enabled and config.training:
        #     ctx.save_for_backward(output, input, input_mask)
        #     ctx.config = config
        return output


# just for layer test
class TransformerEncoderLayer(TransformerEncoderLayerBase):
    """
    Initialize the Lightseq Transformer Encoder Layer.

    Static variable:
        layer_id: The layer-index counter starting from 0 and incrementing by 1 every time a layer object is instantiated,
        e.g. if a model has 24 transformer layers, layer_id goes from 0 to 23.
    Arguments:
        config: An object of TransformerEncoderLayer config, see get_config

        initial_weights: Optional: Only used for unit test

        initial_biases: Optional: Only used for unit test
    """

    layer_id = 0

    def __init__(self, config, initial_weights=None, initial_biases=None):
        super(TransformerEncoderLayer, self).__init__()

        self.config = copy.deepcopy(config)
        self.config.layer_id = TransformerEncoderLayer.layer_id
        TransformerEncoderLayer.layer_id = TransformerEncoderLayer.layer_id + 1

        print("Lightseq Transformer config is ", self.config.__dict__)

        if self.config.local_rank is not None and self.config.local_rank >= 0:
            torch.cuda.set_device(self.config.local_rank)

        self.create_cpp_layer()
        self.assigned_layer_weight_grad = False

        hs = self.config.hidden_size
        ims = self.config.intermediate_size
        self.hs = hs
        self.ims = ims
        self.para_offset = TransformerEncoderLayer.gen_offset(hs, ims)
        self.para = torch.nn.Parameter(torch.Tensor(self.para_offset[-1]))

        if initial_weights is None or initial_biases is None:
            print("error! does not complete.")
            exit(-1)
            # self.init_transformer_weights()
            # return

        # For testing only.
        qkv_w = [ele.detach().clone() for ele in initial_weights[:3]]
        qkv_w = torch.cat(qkv_w, dim=0)
        weights = [qkv_w] + [copy_para(ele) for ele in initial_weights[3:]]

        qkv_b = [ele.detach().clone() for ele in initial_biases[:3]]
        qkv_b = torch.cat(qkv_b, dim=0)
        biases = [qkv_b] + [copy_para(ele) for ele in initial_biases[3:]]

        idx = 0
        for w, b in zip(weights, biases):
            cur_para = self._get_weights(idx)
            assert cur_para.numel() == w.numel()
            cur_para.copy_(w.view(-1))
            idx += 1

            cur_para = self._get_weights(idx)
            assert cur_para.numel() == b.numel()
            cur_para.copy_(b.view(-1))
            idx += 1

        self.to(torch.device("cuda:0"), dtype=torch.half)

        if self.config.fp16 and self.para.dtype != torch.half:
            if hasattr(self, "para_16"):
                self.para_16.copy_(self.para.to(torch.half))
            else:
                self.register_buffer("para_16", self.para.clone().detach().half())

        self.assign_layer_weight_grad()

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
        ]
        offsets = calc_offset(sizes)
        return offsets

    def create_cpp_layer(self):

        # create the layer in cuda kernels.
        cuda_module = cuda_layer_module
        create_layer_func = (
            cuda_module.create_transformer_encoder_layer_new_fp16
            if self.config.fp16
            else cuda_module.create_transformer_encoder_layer_new_fp32
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
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1.0 / math.sqrt(fan_in)
        return bound

    def assign_layer_weight_grad(self):
        if self.assigned_layer_weight_grad == True:
            return
        self.assigned_layer_weight_grad = True
        param = (
            self.para_16
            if self.config.fp16 and self.para.dtype != torch.half
            else self.para
        )
        if self.config.layer_id in _all_layer_grads:
            return
        global transformer_cuda_module
        cuda_module = cuda_layer_module
        if self.config.fp16:
            func = cuda_module.assign_layer_weight_grad_fp16
        else:
            func = cuda_module.assign_layer_weight_grad_fp32
        grad = torch.empty_like(param)
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

        self.config.is_grad_enabled = torch.is_grad_enabled()
        hidden_states = hidden_states.contiguous()
        encoder_padding_mask = (
            (encoder_padding_mask * -1e8).type_as(hidden_states).contiguous()
        )
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
            self.config,
        )

        return output.to(self.para)
