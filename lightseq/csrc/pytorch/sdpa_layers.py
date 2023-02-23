import torch
from csrc.pytorch.builder.cuda_layer_builder import CudaLayerBuilder

cuda_layer_module = CudaLayerBuilder().load()


class SdpaLayerFunc(torch.autograd.Function):
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

        return output
