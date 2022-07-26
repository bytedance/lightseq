import torch.nn.functional as F
from torch.nn import Linear
from lightseq.training.pytorch_quantization.tensor_quant import (
    QuantDescriptor,
    QUANT_DESC_8BIT_PER_TENSOR,
)
from lightseq.training.pytorch_quantization.nn.modules.tensor_quantizer import (
    TensorQuantizer,
    enable_quant,
    disable_quant,
    qat_mode,
    ptq_mode,
)


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
        if pre_activation == "relu":
            input_quant_config = relu_quant_config
        else:
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
        output = F.linear(qinput, qweight)
        if self.output_quant is not None:
            output = self.output_quant(output)
        if self.bias is not None:
            output = output + self.bias

        return output
