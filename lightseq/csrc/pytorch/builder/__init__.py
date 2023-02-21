from .builder import CUDAOpBuilder
from .cuda_kernel_builder import CudaKernelBuilder
from .x86_kernel_builder import X86KernelBuilder

# from .transformer_builder import TransformerBuilder
# from .operator_builder import OperatorBuilder
# from .adam_builder import AdamBuilder
# from .layer_builder import LayerBuilder

# TODO: infer this list instead of hard coded
# List of all available ops
__op_builders__ = [
    # AdamBuilder(),
    CudaKernelBuilder(),
    X86KernelBuilder(),
    # OperatorBuilder(),
    # LayerBuilder(),
    # TransformerBuilder(),
]
ALL_OPS = {op.name: op for op in __op_builders__}
