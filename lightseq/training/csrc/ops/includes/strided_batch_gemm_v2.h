/* Copyright 2021 The LightSeq Team
   Copyright Microsoft DeepSpeed
   This file is adapted from Microsoft DeepSpeed
*/
#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>

#include <array>

#include "cublas_wrappers.h"
#include "int8_kernels.h"

template <typename T>
class StridedBatchGemm {
 public:
  struct Config {
    int m;
    int n;
    int k;

    Config() {}
    void SetConfig(int mm, int nn, int kk) {
      m = mm;
      n = nn;
      k = kk;
    }
  };

  StridedBatchGemm(const Config &config) : _config(config) {}

  virtual ~StridedBatchGemm() {}

  void Forward(int bsz, const T *A, const int8_t *B, int32_t *C,
               int8_t *A_buffer, int32_t *C_buffer, cublasHandle_t handle,
               cudaStream_t stream) {
    int m = _config.m, n = _config.n, k = _config.k;
    int stride_a = m * k;
    int stride_b = n * k;
    int stride_c = m * n;

    float scale_A = 127, scale_B = 127, clip_max_A = 16.0, clip_max_B = 16.0;
    launch_quantize_tensor(A, A_buffer, m * n, scale_A, clip_max_A, stream);

    cublas_strided_batched_gemm(handle, m, n, k, &alpha, &beta, A_buffer, B, C,
                                CUBLAS_OP_N, CUBLAS_OP_N, stride_a, stride_b,
                                stride_c, bsz, cublasGemmAlgo_t(99));
  }

  inline void SetConfig(int m, int n, int k) { _config.SetConfig(m, n, k); }

 private:
  Config _config;
  int32_t alpha = 1, beta = 0;
};
