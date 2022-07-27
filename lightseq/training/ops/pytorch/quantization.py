import torch.nn.functional as F
import torch
from torch.nn import Linear
from torch.autograd import Function
from lightseq.training.pytorch_quantization.tensor_quant import (
    QuantDescriptor,
    QUANT_DESC_8BIT_PER_TENSOR,
    _tensor_quant,
)
from lightseq.training.pytorch_quantization.nn.modules.tensor_quantizer import (
    TensorQuantizer,
    enable_quant,
    disable_quant,
    qat_mode,
    ptq_mode,
)

class LSLinear(Function):
    @staticmethod
    def forward(ctx, inp, weight):

        output = F.linear(inp, weight)

        ctx.save_for_backward(inp, weight)
        return output

    @staticmethod
    def quant_transform(tensor):
        tensor_max = tensor.abs().flatten().topk(100)[0][-1]
        out_dtype = torch.float16
        tensor, scale = _tensor_quant(tensor, tensor_max, 8, False, True)
        tensor = (tensor * scale).to(out_dtype)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        (inp, weight) = ctx.saved_tensors
        out_size, in_size = weight.size()
        bsz, seq_len, _ = inp.size()
        # out_dtype = grad_output.dtype
        # reduce_dim = tuple(range(grad_output.dim() - 1))
        # grad_output_max = grad_output.abs().max()
        # grad_output_max = grad_output.abs().amax(reduce_dim)
        # grad_output_max = grad_output.abs().float().quantile(0.9999)
        # grad_output_max = grad_output.abs().flatten().topk(100)[0][-1]
        # grad_output, scale = _tensor_quant(grad_output, grad_output_max, 8, False, True)
        # grad_output = (grad_output * scale).to(out_dtype)
        grad_output = LSLinear.quant_transform(grad_output)
        weight = LSLinear.quant_transform(weight)
        grad_input = F.linear(grad_output, weight.T)
        grad_input = LSLinear.quant_transform(grad_input)
        inp = LSLinear.quant_transform(inp)

        grad_weight = F.linear(
            grad_output.reshape(-1, out_size).T, inp.reshape(-1, in_size).T
        )
        grad_weight = LSLinear.quant_transform(grad_weight)
        # grad_input_max = grad_input.abs().amax(reduce_dim)
        # # grad_input_max = grad_input.abs().float().quantile(0.9999)
        # grad_input, scale = _tensor_quant(grad_input, grad_input_max, 8, False, True)
        # grad_input = (grad_input * scale).to(out_dtype)

        # grad_weight = LSLinear.quant_transform(grad_weight)
        # grad_weight_max = grad_weight.abs().max()
        # grad_weight_max = grad_weight.abs().amax(-1).unsqueeze(-1)
        # grad_weight_max = grad_weight.abs().float().quantile(0.9999)
        # grad_weight_max = grad_weight.abs().flatten().topk(100)[0][-1]
        # grad_weight, scale = _tensor_quant(grad_weight, grad_weight_max, 8, False, True)
        # grad_weight = (grad_weight * scale).to(out_dtype)
        return (grad_input, grad_weight)


act_quant_config = QuantDescriptor(
    num_bits=8, narrow_range=True, learn_amax=True, amax=16.0
)
out_quant_config = QuantDescriptor(
    num_bits=8, narrow_range=True, learn_amax=False, amax=16.0
)
relu_quant_config = QuantDescriptor(
    num_bits=8, narrow_range=True, learn_amax=True, amax=16.0, unsigned=True
)
weight_quant_config = QuantDescriptor(
    num_bits=8, narrow_range=True, learn_amax=False, amax=1.0
)
emb_quant_config = QuantDescriptor(
    num_bits=8, narrow_range=True, learn_amax=True, amax=1.0
)


class QuantLinear(Linear):
    def __init__(self, in_features, out_features, pre_activation=None, *args, **kwargs):
        super(QuantLinear, self).__init__(in_features, out_features, *args, **kwargs)
        # if pre_activation == "relu":
        #     input_quant_config = relu_quant_config
        # else:
        #     input_quant_config = act_quant_config
        input_quant_config = act_quant_config

        self.input_quant = None
        if pre_activation != "encoder_out":
            self.input_quant = TensorQuantizer(input_quant_config)
        self.output_quant = None
        if pre_activation is None or pre_activation != "encoder_out":
            self.output_quant = TensorQuantizer(out_quant_config)

        self.weight_quant = TensorQuantizer(weight_quant_config)

    def forward(self, input):
        qinput = input
        if self.input_quant is not None:
            qinput = self.input_quant(input)
        qweight = self.weight_quant(self.weight)

        if self.weight_quant._disabled:
            # print('fp16 linear forward')
            output = F.linear(qinput, qweight)
        else:
            # print('quant linear forward')
            output = LSLinear.apply(qinput, qweight)

        if self.output_quant is not None:
            output = self.output_quant(output)
        if self.bias is not None:
            output = output + self.bias

        return output
