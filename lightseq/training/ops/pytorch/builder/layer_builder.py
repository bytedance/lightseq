# Copyright 2021 The LightSeq Team
# Copyright Microsoft DeepSpeed
# This file is adapted from Microsoft DeepSpeed

import torch
import pathlib
from .builder import CUDAOpBuilder
from .builder import installed_cuda_version


class LayerBuilder(CUDAOpBuilder):
    NAME = "lightseq_layers_new"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f"op_builder.{self.NAME}_op"

    def sources(self):
        return [
            "csrc/kernels/cublas_wrappers.cu",
            "csrc/kernels/transform_kernels.cu",
            "csrc/kernels/transform_kernels_new.cu",
            "csrc/kernels/dropout_kernels.cu",
            "csrc/kernels/normalize_kernels.cu",
            "csrc/kernels/softmax_kernels.cu",
            "csrc/kernels/softmax_kernels_new.cu",
            "csrc/kernels/general_kernels.cu",
            "csrc/kernels/cuda_util.cu",
            "csrc/kernels/embedding_kernels.cu",
            "csrc/kernels/cross_entropy.cu",
            "csrc/kernels/crf.cu",
            "csrc/lsflow/context.cpp",
            "csrc/lsflow/layer.cpp",
            "csrc/lsflow/manager.cpp",
            "csrc/lsflow/node.cpp",
            "csrc/lsflow/tensor.cpp",
            "csrc/ops_new/bias_act_dropout.cpp",
            "csrc/ops_new/bias_dropout_residual.cpp",
            "csrc/ops_new/linear.cpp",
            "csrc/ops_new/layer_normalize.cpp",
            "csrc/ops_new/strided_batch_gemm.cpp",
            "csrc/ops_new/bias_add_transform_20314.cpp",
            "csrc/ops_new/dropout.cpp",
            "csrc/ops_new/softmax.cpp",
            "csrc/ops_new/launch_concat3_dim1.cpp",
            "csrc/ops_new/transform_0213.cpp",
            "csrc/ops_new/crf.cpp",
            "csrc/layers_new/feed_forward_layer.cpp",
            "csrc/layers_new/multihead_attention_layer.cpp",
            "csrc/layers_new/transformer_encoder_layer.cpp",
            "csrc/layers_new/dec_self_attention_layer.cpp",
            "csrc/layers_new/encdec_kv_layer.cpp",
            "csrc/layers_new/dec_enc_attention_layer.cpp",
            "csrc/layers_new/transformer_decoder_layer.cpp",
            "csrc/layers_new/crf_layer.cpp",
            "csrc/pybind/pybind_layer_new.cpp",
        ]

    def include_paths(self):
        paths = [
            "csrc/kernels/includes",
            "csrc/ops_new/includes",
            "csrc/lsflow/includes",
            "csrc/layers_new/includes",
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
        return [
            "-O3",
            "-std=c++14",
            "-g",
            "-Wno-reorder",
            "-DPYBIND_LAYER",
            # "-DDEBUG_MODE",
        ]
