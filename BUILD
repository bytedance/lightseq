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
    includes = ["3rdparty/cub-1.8.0"],
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
        "model/encoder.h",
    ]),
    copts = cuda_default_copts(),
    deps = [
        ":transformer_kernel",
        ":transformer_weight",
        ":util",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "gpt_encoder",
    srcs = ["model/gpt_encoder.cu.cc"],
    hdrs = glob([
        "model/gpt_encoder.h",
    ]),
    copts = cuda_default_copts(),
    deps = [
        ":gpt_kernel",
        ":gpt_weight",
        ":transformer_kernel",
        ":util",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "transformer_decoder",
    srcs = ["model/decoder.cu.cc"],
    hdrs = glob([
        "model/decoder.h",
    ]),
    copts = cuda_default_copts(),
    includes = ["3rdparty/cub-1.8.0"],
    deps = [
        ":transformer_kernel",
        ":transformer_weight",
        ":util",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "transformer_server",
    srcs = ["server/transformer_server.cu.cc"],
    copts = cuda_default_copts(),
    deps = [
        "transformer_decoder",
        "transformer_encoder",
        ":transformer_weight",
        ":util",
        "//src/core:model_config",
        "//src/core:model_config_cuda",
        "//src/core:model_config_proto",
        "//src/servables/custom",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_binary(
    name = "libtransformer.so",
    linkopts = ["-pthread"],
    linkshared = 1,
    deps = [
        "transformer_encoder",
        "transformer_proto",
        ":transformer_decoder",
        ":transformer_kernel",
        ":transformer_server",
        ":transformer_weight",
        ":util",
    ],
)

cc_library(
    name = "generate_server",
    srcs = ["server/generate_server.cu.cc"],
    copts = cuda_default_copts(),
    deps = [
        "transformer_decoder",
        "transformer_encoder",
        ":transformer_weight",
        ":util",
        "//src/core:model_config",
        "//src/core:model_config_cuda",
        "//src/core:model_config_proto",
        "//src/servables/custom",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_binary(
    name = "libgenerate.so",
    linkopts = ["-pthread"],
    linkshared = 1,
    deps = [
        "transformer_encoder",
        "transformer_proto",
        ":generate_server",
        ":transformer_decoder",
        ":transformer_kernel",
        ":transformer_weight",
        ":util",
    ],
)

cc_library(
    name = "gptlm_server",
    srcs = ["server/gptlm_server.cu.cc"],
    copts = cuda_default_copts(),
    deps = [
        "gpt_encoder",
        ":gpt_weight",
        ":util",
        "//src/core:model_config",
        "//src/core:model_config_cuda",
        "//src/core:model_config_proto",
        "//src/servables/custom",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_binary(
    name = "libgptlm.so",
    linkopts = ["-pthread"],
    linkshared = 1,
    deps = [
        "gpt_encoder",
        "gpt_proto",
        ":gpt_kernel",
        ":gpt_weight",
        ":gptlm_server",
        ":util",
    ],
)

cc_library(
    name = "gpt_generate_server",
    srcs = ["server/gpt_generate_server.cu.cc"],
    copts = cuda_default_copts(),
    deps = [
        "gpt_encoder",
        ":gpt_weight",
        ":util",
        "//src/core:model_config",
        "//src/core:model_config_cuda",
        "//src/core:model_config_proto",
        "//src/servables/custom",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_binary(
    name = "libgptgenerate.so",
    linkopts = ["-pthread"],
    linkshared = 1,
    deps = [
        "gpt_encoder",
        "gpt_proto",
        ":gpt_generate_server",
        ":gpt_kernel",
        ":gpt_weight",
        ":util",
    ],
)

cc_binary(
    name = "transformer_example",
    srcs = ["example/transformer_example.cu.cc"],
    linkopts = [
        "-L/usr/local/cuda/lib64/stubs",
        "-L/usr/local/cuda/lib64",
        "-pthread",
        "-lcudart",
        "-lcublas",
    ],
    deps = [
        "transformer_decoder",
        "transformer_encoder",
        ":transformer_weight",
        ":util",
    ],
)

cc_binary(
    name = "gptlm_example",
    srcs = ["example/gptlm_example.cu.cc"],
    linkopts = [
        "-L/usr/local/cuda/lib64/stubs",
        "-L/usr/local/cuda/lib64",
        "-pthread",
        "-lcudart",
        "-lcublas",
    ],
    deps = [
        "gpt_encoder",
        ":gpt_weight",
        ":util",
    ],
)

cc_binary(
    name = "gpt_generate_example",
    srcs = ["example/gpt_generation.cu.cc"],
    linkopts = [
        "-L/usr/local/cuda/lib64/stubs",
        "-L/usr/local/cuda/lib64",
        "-pthread",
        "-lcudart",
        "-lcublas",
    ],
    deps = [
        "gpt_encoder",
        ":gpt_weight",
        ":util",
    ],
)
