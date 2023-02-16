# Copyright 2021 The LightSeq Team
# Copyright Microsoft DeepSpeed
# This builder is adapted from Microsoft DeepSpeed

import torch
import pathlib
from .builder import CUDAOpBuilder
from .builder import installed_cuda_version


class CudaKernelBuilder(CUDAOpBuilder):
    NAME = "lightseq_kernels"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f"op_builder.{self.NAME}_op"

    def sources(self):
        return [
            "csrc/kernels/cuda/gemm_test.cpp",
            "csrc/kernels/cuda/cuda_util.cu",
            "csrc/kernels/cuda/transform_kernels.cu",
            "csrc/kernels/cuda/transform_kernels_new.cu",
            "csrc/kernels/cuda/softmax_kernels.cu",
            "csrc/kernels/cuda/softmax_kernels_new.cu",
            "csrc/kernels/cuda/general_kernels.cu",
            "csrc/kernels/cuda/normalize_kernels.cu",
            "csrc/kernels/cuda/dropout_kernels.cu",
            "csrc/kernels/cuda/embedding_kernels.cu",
            "csrc/kernels/cuda/quantize_kernels.cu",
            "csrc/kernels/cuda/crf.cu",
            "csrc/pybind/pybind_kernel_cuda.cpp",
        ]

    def include_paths(self):
        paths = [
            "csrc/kernels/cuda/includes",
        ]
        cuda_major, cuda_minor = installed_cuda_version()
        if cuda_major < 11:
            paths.append(str(pathlib.Path(__file__).parents[5] / "3rdparty" / "cub"))
        return paths

    def nvcc_args(self):
        args = [
            "-O3",
            "--use_fast_math",
            "-std=c++14",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-DTHRUST_IGNORE_CUB_VERSION_CHECK",
        ]

        return args + self.compute_capability_args()

    def cxx_args(self):
        return ["-O3", "-std=c++14", "-g", "-Wno-reorder"]
