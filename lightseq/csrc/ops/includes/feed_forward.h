#pragma once

/* Copyright 2021 The LightSeq Team
   Copyright Microsoft DeepSpeed
   This file is adapted from Microsoft DeepSpeed
*/
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>

#include <array>

#include "cublas_wrappers.h"
#include "kernels.h"

template <typename T>
class FeedForward {
 public:
  struct Config {
    int outputSize;
    int inputSize;
    std::array<int, 3> gemm_algos;
    Config(int outputs, int inputs)
        : outputSize(outputs),
          inputSize(inputs),
          gemm_algos(std::array<int, 3>({99, 99, 99})) {}
  };
  FeedForward(Config config) : config_(config) {}

  ~FeedForward() {}

  void Forward(int bsz, const T *input_ptr, const T *weights, T *out,
               cublasHandle_t &_cublasHandle);

  void Backward(int bsz, const T *out_grad, const T *input_ptr,
                const T *weights, T *weights_grad, T *bias_grad,
                cublasHandle_t &_cublasHandle, cudaStream_t &stream,
                T *inp_grad_out = nullptr, T *out_grad_trans_out = nullptr,
                bool compute_bias = true);

  void reset_size(int outputSize, int inputSize);

 private:
  Config config_;
};
