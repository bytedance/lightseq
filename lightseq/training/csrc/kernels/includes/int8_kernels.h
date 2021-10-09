#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include "cuda_util.h"

#define MAX_THREADS 1024

template <typename T>
void launch_quantize_tensor(const T *input, int8_t *output, int total_count,
                            float scale, float clip_max, cudaStream_t &stream);

template <typename T>
void launch_dequantize_tensor(const int32_t *input, T *output, int total_count,
                              float scale, float clip_max,
                              cudaStream_t &stream);

template <typename T>
void quant_trans_weight(const T *input, int8_t *output, int m, int n,
                        float scale, float clip_max);

template <typename T>
void launch_layer_norm_int8O(int8_t *ln_res, T *vars, T *means, const T *inp,
                             const T *scale, const T *bias, int batch_size,
                             int hidden_dim, float quant_scale, float clip_max,
                             cudaStream_t stream);
