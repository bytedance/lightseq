from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F

from lightseq.training.ops.pytorch.quantization import (
    weight_quant_config,
    act_quant_config,
)
from lightseq.training.ops.pytorch.builder import TransformerBuilder

transformer_cuda_module = None


class LSQuantLinearFunc(Function):
    @staticmethod
    def forward(ctx, config, inputs, weight, clip_max):
        cuda_module = transformer_cuda_module
        forward_func = (
            cuda_module.quant_linear_layer_fw_fp16
            if config.fp16
            else cuda_module.quant_linear_layer_fw_fp32
        )

        if config.fp16:
            inputs = inputs.to(torch.half)

        outputs = forward_func(
            config.layer_id, inputs, weight, clip_max, config.quant_mode
        )

        if config.is_grad_enabled and config.training:
            ctx.save_for_backward(inputs, weight)
            ctx.config = config
        return outputs

    @staticmethod
    def backward(ctx, grad_out):

        assert ctx.config.training

        (inputs, weight) = ctx.saved_tensors
        out_size, in_size = weight.size()
        grad_inputs = F.linear(grad_out, weight.T)
        grad_weight = F.linear(
            grad_out.reshape(-1, out_size).T,
            inputs.reshape(-1, in_size).T,
        )

        return (None, grad_inputs, grad_weight, None)


class LSQuantLinearLayer(nn.Module):
    """Initialize the Lightseq Quant Linear Layer.

    Static variable:
        layer_id: The layer-index counter starting from 0 and incrementing by 1 every time a layer object is instantiated,
    Arguments:
        config: An object of LSQuantLinearLayer config, see get_config
    """

    layer_id = 0

    def __init__(
        self,
        config,
    ):
        super(LSQuantLinearLayer, self).__init__()
        self.config = config
        self.config.layer_id = LSQuantLinearLayer.layer_id
        LSQuantLinearLayer.layer_id += 1

        self.quant_mode = False

        if self.config.local_rank >= 0:
            torch.cuda.set_device(self.config.local_rank)

        self.weight = nn.Parameter(
            torch.empty(
                (self.config.out_features, self.config.in_features),
            )
        )
        if self.config.bias:
            self.bias = nn.Parameter(
                torch.empty(
                    self.config.out_features,
                )
            )
        else:
            self.register_parameter("bias", None)

        self.register_buffer("clip_max", torch.empty((3,)))
        self.reset_parameters()

        # Load cuda modules if needed
        global transformer_cuda_module
        if transformer_cuda_module is None:
            transformer_cuda_module = TransformerBuilder().load()

        # create the layer in cuda kernels.
        cuda_module = transformer_cuda_module
        create_layer_func = (
            cuda_module.create_quant_linear_layer_fp16
            if self.config.fp16
            else cuda_module.create_quant_linear_layer_fp32
        )

        create_layer_func(
            self.config.layer_id,
            self.config.in_features,
            self.config.out_features,
            self.config.max_batch_tokens,
        )

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        act_cmax = act_quant_config.amax.tolist()
        wei_cmax = weight_quant_config.amax.tolist()
        init_clip_max = torch.tensor([act_cmax, wei_cmax, act_cmax])
        self.clip_max.data = init_clip_max
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @staticmethod
    def get_config(**kwargs):
        @dataclass
        class Config:
            max_batch_tokens: int  # max batch token numbers
            in_features: int
            out_features: int
            bias: bool
            fp16: bool  # fp16 presion
            local_rank: int  # rank in local node

        return Config(**kwargs)

    def forward(self, inputs, **kwargs):
        self.config.training = self.training
        self.config.is_grad_enabled = torch.is_grad_enabled()
        self.config.quant_mode = self.quant_mode

        bs, sl = inputs.size()[:2]
        new_bs = None
        if self.quant_mode and (self.config.out_features * bs * sl) % 8 != 0:
            new_bs = (bs // 8 + 1) * 8
            inputs = torch.nn.functional.pad(
                inputs,
                tuple(0 for _ in range((inputs.dim() - 1) * 2)) + (0, new_bs - bs),
            )
        if bs * sl > self.config.max_batch_tokens:
            raise ValueError(
                f"Batch token numbers {bs * sl} exceeds the limit {self.config.max_batch_tokens}."
            )
        out = LSQuantLinearFunc.apply(self.config, inputs, self.weight, self.clip_max)
        if new_bs is not None:
            out = out[:bs, :, :]
        if self.bias is not None:
            out = out + self.bias
        return out

    def disable_quant(self):
        self.quant_mode = False

    def enable_quant(self):
        self.quant_mode = True
