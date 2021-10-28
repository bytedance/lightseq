#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>

#include "3rdparty/cub/cub/cub.cuh"

namespace lightseq {
namespace cuda {

template <typename T>
void launch_quantize_tensor(const T *input, int8_t *output, int total_count,
                            float scale, float clip_max, cudaStream_t &stream);

template <typename T>
void launch_dequantize_tensor(const int32_t *input, T *output, int total_count,
                              float scale, float clip_max,
                              cudaStream_t &stream);

template <typename T>
void ker_norm_layer_resual_int8O_launcher(int token_num, int hidden_size,
                                          cudaStream_t stream, T *input,
                                          int8_t *output, const T *scale,
                                          const T *bias, const T *residual_bias,
                                          const int max_thread_per_block,
                                          float quant_scale, float clip_max,
                                          bool is_post_ln = false);

}  // namespace cuda
}  // namespace lightseq
