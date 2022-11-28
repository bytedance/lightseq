# Copyright 2021 The LightSeq Team
# Copyright Microsoft DeepSpeed
# This builder is adapted from Microsoft DeepSpeed

import torch
import pathlib
import os
from .builder import CUDAOpBuilder


class KernelBuilder(CUDAOpBuilder):
    NAME = "lightseq_kernels"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f"op_builder.{self.NAME}_op"

    def sources(self):
        if os.getenv('ROCM_PATH') is None:
            return [
                "csrc/kernels/cuda_util.cu",
                "csrc/kernels/transform_kernels.cu",
                "csrc/kernels/softmax_kernels.cu",
                "csrc/kernels/general_kernels.cu",
                "csrc/kernels/normalize_kernels.cu",
                "csrc/kernels/dropout_kernels.cu",
                "csrc/kernels/embedding_kernels.cu",
                "csrc/torch/pybind_kernel.cpp",
            ]
        else:
            return [
            "lightseq/training/csrc/kernels/cuda_util.cu",
            "lightseq/training/csrc/kernels/transform_kernels.cu",
            "lightseq/training/csrc/kernels/softmax_kernels.cu",
            "lightseq/training/csrc/kernels/general_kernels.cu",
            "lightseq/training/csrc/kernels/normalize_kernels.cu",
            "lightseq/training/csrc/kernels/dropout_kernels.cu",
            "lightseq/training/csrc/kernels/embedding_kernels.cu",
            "lightseq/training/csrc/torch/pybind_kernel.cpp",
        ]

    def include_paths(self):
        include_file_list = [os.path.abspath("lightseq/training/csrc/kernels/includes"), os.path.abspath("lightseq/training/csrc/ops/includes")]
        if os.getenv('ROCM_PATH') is None:
                include_file_list.append(str(pathlib.Path(__file__).parents[5] / "3rdparty" / "cub"),)
        return include_file_list

    def nvcc_args(self):
        if os.getenv('ROCM_PATH') is not None: 
            args = [
                "-O3",
                "-U__HIP_NO_HALF_OPERATORS__",
                "-U__HIP_NO_HALF_CONVERSIONS__",
                "-U__HIP_NO_HALF2_OPERATORS__",
                "-DTHRUST_IGNORE_CUB_VERSION_CHECK",
                "--gpu-max-threads-per-block=1024",
            ]
        else:
            args = [
                "-O3",
                "--use_fast_math",
                "-std=c++14",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-DTHRUST_IGNORE_CUB_VERSION_CHECK",
            ]
        if os.getenv('ROCM_PATH') is not None:
            return args 
        else:
            return args + self.compute_capability_args()

    def cxx_args(self):
        return ["-O3", "-std=c++14", "-g", "-Wno-reorder"]
