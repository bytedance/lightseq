# Copyright 2021 The LightSeq Team
# Copyright Microsoft DeepSpeed
# This builder is adapted from Microsoft DeepSpeed

import torch
import pathlib
import os
from .builder import OpBuilder
from .builder import installed_cuda_version


class X86KernelBuilder(OpBuilder):
    NAME = "lightseq_kernels"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f"op_builder.{self.NAME}_op"

    def sources(self):
        return [
            "csrc/kernels/x86/util.cc",
            "csrc/kernels/x86/gemm.cpp",
            "csrc/pybind/pybind_kernel_x86.cpp",
        ]

    def include_paths(self):
        paths = [
            "csrc/kernels/x86/includes",
            "/opt/intel/oneapi/mkl/latest/include/",
        ]
        cuda_major, cuda_minor = installed_cuda_version()
        if cuda_major < 11:
            paths.append(str(pathlib.Path(__file__).parents[5] / "3rdparty" / "cub"))
            paths.append(
                str(pathlib.Path(__file__).parents[5] / "3rdparty" / "pybind11")
            )
        return paths

    # def library_dirs(self):
    #     return [
    #         "/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/",
    #     ]

    def dynamic_libraries(self):
        # TODO: need to support dynamic lib
        return ["pthread", "m", "dl"]

    def static_libraries(self):
        mkl_root = pathlib.Path(os.getenv("MKLROOT", "/opt/intel/oneapi/mkl/latest/"))
        iomp_root = pathlib.Path(
            os.getenv(
                "IOMPROOT",
                "/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/",
            )
        )
        if not mkl_root.exists:
            raise ValueError(
                "Can't find MKL root, please set MKLROOT or install, default /opt/intel/oneapi/mkl/latest/"
            )
        if not mkl_root.exists:
            raise ValueError(
                "Can't find MKL root, please set IOMPROOT or install, "
                "default /opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/"
            )

        return [
            str(mkl_root.joinpath("lib/intel64/libmkl_intel_ilp64.a")),
            str(mkl_root.joinpath("lib/intel64/libmkl_gnu_thread.a")),
            str(mkl_root.joinpath("lib/intel64/libmkl_core.a")),
        ]

    def nvcc_args(self):
        args = [
            "-O3",
            "--use_fast_math",
            "-std=c++14",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-DTHRUST_IGNORE_CUB_VERSION_CHECK",
            "-Xcompiler=-fopenmp",
        ]

        return args

    def cxx_args(self):
        return [
            "-O3",
            "-std=c++14",
            "-g",
            "-Wno-reorder",
            "-DMKL_ILP64",
            "-m64",
            "-fopenmp",
        ]
