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

template <typename T>
class StridedBatchGemm {
 public:
  struct Config {
    int m;
    int n;
    int k;
    float alpha;
    float beta;
    cublasOperation_t op_A;
    cublasOperation_t op_B;
    std::array<int, 3> gemm_algos;

    Config(float param_alpha, float param_beta, cublasOperation_t opA,
           cublasOperation_t opB)
        : alpha(param_alpha),
          beta(param_beta),
          op_A(opA),
          op_B(opB),
          gemm_algos(std::array<int, 3>({99, 99, 99})) {}
    void SetConfig(int mm, int nn, int kk) {
      m = mm;
      n = nn;
      k = kk;
    }
  };

  StridedBatchGemm(const Config &config) : _config(config) {}

  virtual ~StridedBatchGemm() {}

  void Forward(int bsz, T *output, const T *_buffer_a, const T *_buffer_b,
               cublasHandle_t handle);

  void Backward(int bsz, const T *d_output, const T *_buffer_a,
                const T *_buffer_b, cublasHandle_t handle, T *inpGradA = nullptr,
                T *inpGradB = nullptr);

  void SetConfig(int m, int n, int k);

 private:
  Config _config;
};
