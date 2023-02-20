import torch

from builder.cuda_layer_builder import CudaLayerBuilder

cuda_layer_module = CudaLayerBuilder().load()


class LSTransformerEncoderFuncNew(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        input_mask,
        parameters,
        config,
    ):
        forward_func = (
            cuda_layer_module.transformer_encoder_layer_fw_fp16
            if config.fp16
            else cuda_layer_module.transformer_encoder_layer_fw_fp32
        )
        if config.fp16:
            input = input.to(torch.half)
            input_mask = input_mask.to(torch.half)

        (output,) = forward_func(config.layer_id, input, input_mask)

        if config.is_grad_enabled and config.training:
            ctx.save_for_backward(output, input, input_mask)
            ctx.config = config
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert ctx.config.training

        backward_func = (
            cuda_layer_module.transformer_encoder_layer_bw_fp16
            if ctx.config.fp16
            else cuda_layer_module.transformer_encoder_layer_bw_fp32
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

        # grad = _all_layer_grads[ctx.config.layer_id]

        # return (grad_input, None, grad, None)


if __name__ == "__main__":
    pass
