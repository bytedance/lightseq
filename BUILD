# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package(
    default_visibility = ["//visibility:public"],
)

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts")
load("@protobuf_archive//:protobuf.bzl", "cc_proto_library")

cc_proto_library(
    name = "transformer_proto",
    srcs = ["proto/transformer.proto"],
)

cc_library(
    name = "transformer_kernel",
    srcs = ["kernels/nmtKernels.cu.cc"],
    hdrs = glob([
        "kernels/*.h",
    ]),
    copts = cuda_default_copts(),
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "transformer_util",
    srcs = ["util.cu.cc"],
    hdrs = ["util.h"],
    copts = cuda_default_copts(),
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "transformer_weight",
    srcs = ["proto/transformer_weight.cu.cc"],
    hdrs = glob([
        "proto/transformer_weight.h",
    ]),
    copts = cuda_default_copts(),
    deps = [
        ":transformer_proto",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "transformer_encoder",
    srcs = ["model/encoder.cu.cc"],
    hdrs = glob([
        "model/encoder.h"
    ]),
    copts = cuda_default_copts(),
    deps = [
        ":transformer_weight",
       ":transformer_util",
       ":transformer_kernel",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "transformer_decoder",
    srcs = ["model/decoder.cu.cc"],
    hdrs = glob([
        "model/decoder.h"
    ]),
    copts = cuda_default_copts(),
    includes = ["cub-1.8.0"],
    deps = [
        ":transformer_weight",
       ":transformer_util",
       ":transformer_kernel",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "transformer_base",
    srcs = ["transformer.cu.cc"],
    copts = cuda_default_copts(),
    deps = [
        ":transformer_weight",
        ":transformer_util",
        "transformer_encoder",
        "transformer_decoder",
        "//src/core:model_config",
        "//src/core:model_config_cuda",
        "//src/core:model_config_proto",
        "//src/servables/custom:custom",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_binary(
    name = "libtransformer.so",
    deps = [
        ":transformer_base",
        "transformer_proto",
        ":transformer_kernel",
        ":transformer_weight",
       ":transformer_util",
        "transformer_encoder",
        ":transformer_decoder",
    ],
    linkopts = ["-pthread"],
    linkshared = 1,
)

cc_library(
    name = "generate_base",
    srcs = ["generate.cu.cc"],
    copts = cuda_default_copts(),
    deps = [
        ":transformer_weight",
        ":transformer_util",
        "transformer_encoder",
        "transformer_decoder",
        "//src/core:model_config",
        "//src/core:model_config_cuda",
        "//src/core:model_config_proto",
        "//src/servables/custom:custom",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_binary(
    name = "libgenerate.so",
    deps = [
        ":generate_base",
        "transformer_proto",
        ":transformer_kernel",
        ":transformer_weight",
       ":transformer_util",
        "transformer_encoder",
        ":transformer_decoder",
    ],
    linkopts = ["-pthread"],
    linkshared = 1,
)

cc_binary(
    name = "example",
    srcs = ["example.cu.cc"],
    deps = [
        ":transformer_weight",
        ":transformer_util",
        "transformer_encoder",
        "transformer_decoder",        
    ],
    linkopts = [
        "-L/usr/local/cuda/lib64/stubs",
        "-L/usr/local/cuda/lib64",
        "-pthread",
        "-lcudart",
        "-lcublas"
    ],
)
