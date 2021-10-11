#pragma once

/* Copyright 2021 The LightSeq Team
   Copyright Microsoft DeepSpeed
   This file is adapted from Microsoft DeepSpeed
*/
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>

#include <array>

#include "cuda_util.h"
#include "cublas_wrappers.h"
#include "kernels.h"
#include "int8_kernels.h"

template <typename T>
class FeedForwardV4 {
 public:
  struct Config {
    int m, n, k;
    Config() {}
    void SetConfig(int mm, int nn, int kk) {
      m = mm;
      n = nn;
      k = kk;
    }
  };

  FeedForwardV4(Config config) : _config(config) {}

  ~FeedForwardV4() {}

  void Forward(const int8_t *A, const T *B, T *C, int8_t *B_buffer,
               int32_t *C_buffer, cublasHandle_t handle, cudaStream_t stream) {
    int m = _config.m, n = _config.n, k = _config.k;
    int align = 16;
    n = (n + align - 1) / align * align;

    float scale_A = 127, scale_B = 127, clip_max_A = 0.3, clip_max_B = 16.0;
    launch_quantize_tensor(B, B_buffer, n * k, scale_B, clip_max_B, stream);

    cublas_gemm_ex(handle, CUBLAS_OP_T, CUBLAS_OP_N, _config.m, n, _config.k,
                   &alpha, &beta, A, B_buffer, C_buffer, cublasGemmAlgo_t(99));

    launch_dequantize_tensor(C_buffer, C, m * _config.n, scale_A * scale_B,
                             clip_max_A * clip_max_B, stream);
  }

  void ForwardV2(const int8_t *A, const int8_t *B, T *C, int8_t *B_buffer,
                 int32_t *C_buffer, cublasHandle_t handle,
                 cudaStream_t stream) {
    int m = _config.m, n = _config.n, k = _config.k;
    int align = 16;
    n = (n + align - 1) / align * align;

    float scale_A = 127, scale_B = 127, clip_max_A = 0.3, clip_max_B = 16.0;

    cublas_gemm_ex(handle, CUBLAS_OP_T, CUBLAS_OP_N, _config.m, n, _config.k,
                   &alpha, &beta, A, B, C_buffer, cublasGemmAlgo_t(99));

    launch_dequantize_tensor(C_buffer, C, m * _config.n, scale_A * scale_B,
                             clip_max_A * clip_max_B, stream);
  }

  void ForwardV3(const int8_t *A, const int8_t *B, int32_t *C, int8_t *B_buffer,
                 int32_t *C_buffer, cublasHandle_t handle,
                 cudaStream_t stream) {
    int m = _config.m, n = _config.n, k = _config.k;
    int align = 16;
    n = (n + align - 1) / align * align;

    cublas_gemm_ex(handle, CUBLAS_OP_T, CUBLAS_OP_N, _config.m, n, _config.k,
                   &alpha, &beta, A, B, C, cublasGemmAlgo_t(99));
  }

  void ForwardV4(const int8_t *A, const T *B, int32_t *C, int8_t *B_buffer,
                 int32_t *C_buffer, cublasHandle_t handle,
                 cudaStream_t stream) {
    int m = _config.m, n = _config.n, k = _config.k;
    int align = 16;
    n = (n + align - 1) / align * align;

    float scale_A = 127, scale_B = 127, clip_max_A = 0.3, clip_max_B = 16.0;
    launch_quantize_tensor(B, B_buffer, n * k, scale_B, clip_max_B, stream);

    cublas_gemm_ex(handle, CUBLAS_OP_T, CUBLAS_OP_N, _config.m, n, _config.k,
                   &alpha, &beta, A, B_buffer, C, cublasGemmAlgo_t(99));
  }

  inline void SetConfig(int m, int n, int k) { _config.SetConfig(m, n, k); }

 private:
  Config _config;
  int32_t alpha = 1, beta = 0;
};
