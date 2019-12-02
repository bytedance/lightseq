# Copyright (c) 2019, ByteDance CORPORATION. All rights reserved.

package(
    default_visibility = ["//visibility:public"],
)

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts")
load("@protobuf_archive//:protobuf.bzl", "cc_proto_library")

cc_proto_library(
    name = "transformer_proto",
    srcs = ["proto/transformer.proto"],
)

cc_proto_library(
    name = "gpt_proto",
    srcs = ["proto/gpt.proto"],
)

cc_library(
    name = "transformer_kernel",
    srcs = ["kernels/transformerKernels.cu.cc"],
    hdrs = glob([
        "kernels/common.h",
        "kernels/transformerKernels.h",
    ]),
    copts = cuda_default_copts(),
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "gpt_kernel",
    srcs = ["kernels/gptKernels.cu.cc"],
    hdrs = glob([
        "kernels/common.h",
        "kernels/gptKernels.h",
        "kernels/transformerKernels.h",
    ]),
    copts = cuda_default_copts(),
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "util",
    srcs = ["tools/util.cu.cc"],
    hdrs = ["tools/util.h"],
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
        ":util",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "gpt_weight",
    srcs = ["proto/gpt_weight.cu.cc"],
    hdrs = glob([
        "proto/gpt_weight.h",
    ]),
    copts = cuda_default_copts(),
    deps = [
        ":gpt_proto",
        ":util",
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
       ":util",
       ":transformer_kernel",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "gpt_encoder",
    srcs = ["model/gpt_encoder.cu.cc"],
    hdrs = glob([
        "model/gpt_encoder.h"
    ]),
    copts = cuda_default_copts(),
    deps = [
        ":gpt_weight",
       ":util",
       ":transformer_kernel",
       ":gpt_kernel",
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
    includes = ["3rdparty/cub-1.8.0"],
    deps = [
        ":transformer_weight",
       ":util",
       ":transformer_kernel",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "transformer_server",
    srcs = ["server/transformer_server.cu.cc"],
    copts = cuda_default_copts(),
    deps = [
        ":transformer_weight",
        ":util",
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
        ":transformer_server",
        "transformer_proto",
        ":transformer_kernel",
        ":transformer_weight",
       ":util",
        "transformer_encoder",
        ":transformer_decoder",
    ],
    linkopts = ["-pthread"],
    linkshared = 1,
)

cc_library(
    name = "generate_server",
    srcs = ["server/generate_server.cu.cc"],
    copts = cuda_default_copts(),
    deps = [
        ":transformer_weight",
        ":util",
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
        ":generate_server",
        "transformer_proto",
        ":transformer_kernel",
        ":transformer_weight",
       ":util",
        "transformer_encoder",
        ":transformer_decoder",
    ],
    linkopts = ["-pthread"],
    linkshared = 1,
)

cc_library(
    name = "gptlm_server",
    srcs = ["server/gptlm_server.cu.cc"],
    copts = cuda_default_copts(),
    deps = [
        ":gpt_weight",
        ":util",
        "gpt_encoder",
        "//src/core:model_config",
        "//src/core:model_config_cuda",
        "//src/core:model_config_proto",
        "//src/servables/custom:custom",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_binary(
    name = "libgptlm.so",
    deps = [
        ":gptlm_server",
        "gpt_proto",
        ":gpt_kernel",
        ":gpt_weight",
       ":util",
        "gpt_encoder",
    ],
    linkopts = ["-pthread"],
    linkshared = 1,
)

cc_binary(
    name = "transformer_example",
    srcs = ["example/transformer_example.cu.cc"],
    deps = [
        ":transformer_weight",
        ":util",
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

cc_binary(
    name = "gptlm_example",
    srcs = ["example/gptlm_example.cu.cc"],
    deps = [
        ":gpt_weight",
        ":util",
        "gpt_encoder",
    ],
    linkopts = [
        "-L/usr/local/cuda/lib64/stubs",
        "-L/usr/local/cuda/lib64",
        "-pthread",
        "-lcudart",
        "-lcublas"
    ],
)
