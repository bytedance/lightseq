#pragma once

/* Copyright 2021 The LightSeq Team
   Copyright Microsoft DeepSpeed
   This file is adapted from Microsoft DeepSpeed
*/
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>

#include <array>

#include "cublas_algo_map.h"
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
               cublasHandle_t &_cublasHandle) {
    float alpha = T(1.);
    float beta = T(0.);

    cublas_gemm_ex(_cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, config_.outputSize,
                   bsz, config_.inputSize, &alpha, &beta, weights, input_ptr,
                   out, cublasGemmAlgo_t(config_.gemm_algos[0]));
  }

  void Forward(int bsz, const int8_t *qinput_ptr, const int8_t *qweight_ptr,
               const float *alpha_ptr, const float *beta_ptr, int8_t *qout_ptr,
               cublasLtHandle_t &cublasLt_handle, cudaStream_t &stream,
               bool col32 = false) {
    if (col32) {
      cublasLtMM_withAlgo_i8IO(qout_ptr, 1, bsz, config_.outputSize,
                               config_.inputSize, 0, 0, 0, alpha_ptr, beta_ptr,
                               qinput_ptr, qweight_ptr, cublasLt_handle, stream,
                               true);
    } else {
      cublaslt_igemm<int8_t, float>(qweight_ptr, qinput_ptr, qout_ptr, 1,
                                    config_.outputSize, bsz, config_.inputSize,
                                    0, 0, 0, alpha_ptr, beta_ptr,
                                    cublasLt_handle, stream);
    }
  }

  void Forward(int bsz, const int8_t *qinput_ptr, const int8_t *qweight_ptr,
               const float *alpha_ptr, const float *beta_ptr, int8_t *qout_ptr,
               cublasLtHandle_t &cublasLt_handle, cudaStream_t &stream,
               cublasLtMatmulAlgo_info &algo_info, cublasAlgoMap &algo_map) {
    if (algo_info.dataOrder != "CUBLASLT_ORDER_COL") {
      cublasLtMM_withAlgo_i8IO(qout_ptr, 1, bsz, config_.outputSize,
                               config_.inputSize, 0, 0, 0, alpha_ptr, beta_ptr,
                               qinput_ptr, qweight_ptr, cublasLt_handle, stream,
                               algo_info, algo_map);
    } else {
      cublaslt_igemm<int8_t, float>(
          qweight_ptr, qinput_ptr, qout_ptr, 1, config_.outputSize, bsz,
          config_.inputSize, 0, 0, 0, alpha_ptr, beta_ptr, cublasLt_handle,
          stream, algo_info, algo_map);
    }
  }

  void Backward(int bsz, const T *out_grad, const T *input_ptr,
                const T *weights, T *weights_grad, T *bias_grad,
                cublasHandle_t &_cublasHandle, cudaStream_t &stream,
                T *inp_grad_out = nullptr, T *out_grad_trans_out = nullptr,
                bool compute_bias = true) {
    float alpha = (T)1.0, beta = (T)0.0;
    cublas_gemm_ex(_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, config_.inputSize,
                   config_.outputSize, bsz, &alpha, &beta, input_ptr, out_grad,
                   weights_grad, cublasGemmAlgo_t(config_.gemm_algos[1]));

    cublas_gemm_ex(_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, config_.inputSize,
                   bsz, config_.outputSize, &alpha, &beta, weights, out_grad,
                   inp_grad_out, cublasGemmAlgo_t(config_.gemm_algos[2]));
    if (compute_bias) {
      launch_fuse_transpose_bias_kernel<T>(out_grad, bias_grad, bsz,
                                           config_.outputSize, stream);
    }
  }

  void reset_size(int outputSize, int inputSize) {
    config_.outputSize = outputSize;
    config_.inputSize = inputSize;
  }

 private:
  Config config_;
};
