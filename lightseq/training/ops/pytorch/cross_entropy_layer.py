from dataclasses import dataclass

import torch
from torch import nn
from torch.autograd import Function

from lightseq.training.ops.pytorch.builder import TransformerBuilder

transformer_cuda_module = None


class LSCrossEntropyFunc(Function):
    @staticmethod
    def forward(ctx, config, inputs, targets):
        cuda_module = transformer_cuda_module
        forward_func = (
            cuda_module.cross_entropy_layer_fw_fp16
            if config.fp16
            else cuda_module.cross_entropy_layer_fw_fp32
        )

        targets = targets.to(torch.int32)
        if config.fp16:
            inputs = inputs.to(torch.half)

        (reduced_loss, nll_loss) = forward_func(config.layer_id, inputs, targets)

        if config.is_grad_enabled and config.training:
            ctx.save_for_backward(inputs, targets)
            ctx.config = config
        return reduced_loss, nll_loss

    @staticmethod
    def backward(ctx, grad_loss, grad_nll_loss):
        cuda_module = transformer_cuda_module
        backward_func = (
            cuda_module.cross_entropy_layer_bw_fp16
            if ctx.config.fp16
            else cuda_module.cross_entropy_layer_bw_fp32
        )

        assert ctx.config.training

        (inputs, targets) = ctx.saved_tensors

        targets = targets.to(torch.int32)
        grad_loss = grad_loss.to(torch.float32)
        if ctx.config.fp16:
            inputs = inputs.to(torch.half)

        (grad_inputs,) = backward_func(ctx.config.layer_id, grad_loss, inputs, targets)

        return (None, grad_inputs, None)


class LSCrossEntropyLayer(nn.Module):
    """Initialize the Lightseq Cross Entropy Layer.

    Static variable:
        layer_id: The layer-index counter starting from 0 and incrementing by 1 every time a layer object is instantiated,
    Arguments:
        config: An object of LSCrossEntropyLayer config, see get_config
    """

    layer_id = 0

    def __init__(
        self,
        config,
    ):
        super(LSCrossEntropyLayer, self).__init__()
        self.config = config
        self.config.layer_id = LSCrossEntropyLayer.layer_id
        LSCrossEntropyLayer.layer_id += 1

        if self.config.local_rank >= 0:
            torch.cuda.set_device(self.config.local_rank)

        # Load cuda modules if needed
        global transformer_cuda_module
        if transformer_cuda_module is None:
            transformer_cuda_module = TransformerBuilder().load()

        # create the layer in cuda kernels.
        cuda_module = transformer_cuda_module
        create_layer_func = (
            cuda_module.create_cross_entropy_layer_fp16
            if self.config.fp16
            else cuda_module.create_cross_entropy_layer_fp32
        )

        create_layer_func(
            self.config.layer_id,
            self.config.epsilon,
            self.config.padding_idx,
            self.config.max_batch_tokens,
        )

    @staticmethod
    def get_config(**kwargs):
        @dataclass
        class Config:
            max_batch_tokens: int  # max batch token numbers
            padding_idx: int  # padding token id in vocabulary
            epsilon: float  # label smoothing factor
            fp16: bool  # fp16 presion
            local_rank: int  # rank in local node

        return Config(**kwargs)

    def forward(self, inputs, targets, **kwargs):
        self.config.training = self.training
        self.config.is_grad_enabled = torch.is_grad_enabled()
        bs, sl = inputs.size()[:2]
        if bs * sl > self.config.max_batch_tokens:
            raise ValueError(
                f"Batch token numbers {bs * sl} exceeds the limit {self.config.max_batch_tokens}."
            )
        loss, nll_loss = LSCrossEntropyFunc.apply(
            self.config, inputs, targets, **kwargs
        )
        return loss, nll_loss
