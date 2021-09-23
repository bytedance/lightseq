#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>

template <typename T>
void launch_quantize_tensor(const T *input, int8_t *output, int total_count,
                            float scale, float clip_max, cudaStream_t &stream);

template <typename T>
void launch_dequantize_tensor(const int32_t *input, T *output, int total_count,
                              float scale, float clip_max,
                              cudaStream_t &stream);
