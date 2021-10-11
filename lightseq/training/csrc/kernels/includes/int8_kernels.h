#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include "cuda_util.h"
#include "kernels.h"

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

template <ActivationType, typename T>
void launch_ls_dropout_act_bias_int32I_int8O(
    int8_t *out, const int32_t *vals, uint8_t *mask, const T *bias,
    int total_count, int dim, float ratio, float in_scale, float in_clip_max,
    float out_scale, float out_clip_max, cudaStream_t stream);

// [b, s, 3, h] -> [3, b, nh, s, ad]
template <typename T>
void launch_bias_add_transform_20314_int32I(T *output, const int32_t *input,
                                            const T *bias, int dim_0, int dim_1,
                                            int dim_2, int dim_3, int dim_4,
                                            float scale, float clip_max,
                                            cudaStream_t stream);

template <typename T>
void launch_ls_dropout_res_bias_int32I(T *out, const int32_t *vals,
                                       uint8_t *mask, const T *bias,
                                       const T *residual, int total_count,
                                       int dim, float ratio, float scale,
                                       float clip_max, cudaStream_t stream);
