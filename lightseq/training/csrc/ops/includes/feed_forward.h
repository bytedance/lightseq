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
               cublasHandle_t &_cublasHandle) {
    float alpha = T(1.);
    float beta = T(0.);

    cublas_gemm_ex(_cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, config_.outputSize,
                   bsz, config_.inputSize, &alpha, &beta, weights, input_ptr,
                   out, cublasGemmAlgo_t(config_.gemm_algos[0]));
  }

  void Forward(int bsz, const int8_t *qinput_ptr, const int8_t *qweight_ptr,
               const float *alpha_ptr, const float *beta_ptr, int8_t *qout_ptr,
               cublasLtHandle_t &cublasLt_handle, cudaStream_t &stream) {
    // launch_quantize<T>(q_input_, clip_mask_, input_ptr, clip_max_,
    //                    bsz * config_.inputSize, input_bit_, stream);
    // launch_quantize<T>(q_weight_, clip_mask_, weights, clip_max_ + 1,
    //                    config_.inputSize * config_.outputSize, weight_bit_,
    //                    stream);
    cublaslt_igemm<int8_t, float>(qweight_ptr, qinput_ptr, qout_ptr, 1,
                                  config_.outputSize, bsz, config_.inputSize, 0,
                                  0, 0, alpha_ptr, beta_ptr, cublasLt_handle,
                                  stream);
    // launch_dequantize<T>(out, clip_mask_, q_output_, clip_max_ + 2,
    //                      bsz * config_.outputSize, output_bit_, stream);
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

  // void set_quant_input(int8_t *q_input) { q_input_ = q_input; }

  // void set_quant_weight(int8_t *q_weight) { q_weight_ = q_weight; }

  // void set_quant_output(int8_t *q_output) { q_output_ = q_output; }

  // void set_clip_mask(uint8_t *clip_mask) { clip_mask_ = clip_mask; }

  // void set_clip_mask_bit(int input_bit, int weight_bit, int output_bit) {
  //   input_bit_ = input_bit;
  //   weight_bit_ = weight_bit;
  //   output_bit_ = output_bit;
  // }

  // void set_clip_max(const T *clip_max) { clip_max_ = clip_max; }

 private:
  Config config_;
  // int8_t *q_input_, *q_weight_, *q_output_;
  // int input_bit_ = 1, weight_bit_ = 3, output_bit_ = 5;
  // uint8_t *clip_mask_;
  // const T *clip_max_;
};
