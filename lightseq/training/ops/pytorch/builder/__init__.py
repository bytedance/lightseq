from .builder import CUDAOpBuilder
from .kernel_builder import KernelBuilder
from .transformer_builder import TransformerBuilder
from .operator_builder import OperatorBuilder
from .adam_builder import AdamBuilder

# TODO: infer this list instead of hard coded
# List of all available ops
__op_builders__ = [
    KernelBuilder(),
    OperatorBuilder(),
    TransformerBuilder(),
    AdamBuilder(),
]
ALL_OPS = {op.name: op for op in __op_builders__}
