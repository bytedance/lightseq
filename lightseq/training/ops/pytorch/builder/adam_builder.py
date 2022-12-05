# Copyright 2021 The LightSeq Team
# Copyright Microsoft DeepSpeed
# This builder is adapted from Microsoft DeepSpeed

import torch
import os
from .builder import CUDAOpBuilder


class AdamBuilder(CUDAOpBuilder):
    NAME = "adam"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f"op_builder.{self.NAME}_op"

    def sources(self):
        if os.getenv("ROCM_PATH") is None:
            return [
                "csrc/kernels/fused_adam_kernel.cu",
                "csrc/torch/pybind_adam.cpp",
            ]
        else:
            return [
                "lightseq/training/csrc/kernels/fused_adam_kernel.cu",
                "lightseq/training/csrc/torch/pybind_adam.cpp",
            ]

    def include_paths(self):
        return [
            os.path.abspath("lightseq/training/csrc/kernels/includes"),
            os.path.abspath("lightseq/training/csrc/ops/includes"),
        ]

    def nvcc_args(self):
        if os.getenv("ROCM_PATH") is not None:
            args = [
                "-O3",
                "-U__HIP_NO_HALF_OPERATORS__",
                "-U__HIP_NO_HALF_CONVERSIONS__",
                "-U__HIP_NO_HALF2_OPERATORS__",
                "-DTHRUST_IGNORE_CUB_VERSION_CHECK",
                "--gpu-max-threads-per-block=1024",
            ]
            return args
        else:
            args = [
                "-O3",
                "--use_fast_math",
                "-std=c++14",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
            ]
            return args + self.compute_capability_args()

    def cxx_args(self):
        return ["-O3", "-std=c++14", "-g", "-Wno-reorder"]
