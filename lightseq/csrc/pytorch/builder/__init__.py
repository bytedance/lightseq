from .builder import CUDAOpBuilder
from .cuda_kernel_builder import CudaKernelBuilder
from .x86_kernel_builder import X86KernelBuilder
from .cuda_layer_builder import CudaLayerBuilder

# TODO: infer this list instead of hard coded
# List of all available ops
__op_builders__ = [
    CudaKernelBuilder(),
    CudaLayerBuilder(),
    X86KernelBuilder(),
]

ALL_OPS = {op.name: op for op in __op_builders__}
