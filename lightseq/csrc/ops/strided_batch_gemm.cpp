/* Copyright 2021 The LightSeq Team
   Copyright Microsoft DeepSpeed
   This file is adapted from Microsoft DeepSpeed
*/
#include "strided_batch_gemm.h"

template <typename T>
void StridedBatchGemm<T>::Forward(int bsz, T *output, const T *_buffer_a,
                                  const T *_buffer_b, cublasHandle_t handle) {
  int stride_a = _config.m * _config.k;
  int stride_b = _config.n * _config.k;
  int stride_c = _config.m * _config.n;

  cublas_strided_batched_gemm(
      handle, _config.m, _config.n, _config.k, &_config.alpha, &_config.beta,
      _buffer_a, _buffer_b, output, _config.op_A, _config.op_B, stride_a,
      stride_b, stride_c, bsz, cublasGemmAlgo_t(_config.gemm_algos[0]));
}

template <typename T>
void StridedBatchGemm<T>::Backward(int bsz, const T *d_output,
                                   const T *_buffer_a, const T *_buffer_b,
                                   cublasHandle_t handle, T *inpGradA,
                                   T *inpGradB) {
  int mb = (_config.op_A == CUBLAS_OP_T ? _config.k : _config.m);
  int kb = (_config.op_A == CUBLAS_OP_T ? _config.m : _config.k);

  int stride_a = mb * _config.n;
  int stride_b = _config.n * kb;
  int stride_c = _config.m * _config.k;

  // B need to transpose.
  cublasOperation_t op_b =
      (_config.op_B == CUBLAS_OP_T ? CUBLAS_OP_N : CUBLAS_OP_T);

  // Calculate d_A.
  cublas_strided_batched_gemm(
      handle, mb, kb, _config.n, &_config.alpha, &_config.beta,
      (_config.op_A == CUBLAS_OP_T ? _buffer_b : d_output),
      (_config.op_A == CUBLAS_OP_T ? d_output : _buffer_b), inpGradA,
      CUBLAS_OP_N, op_b, stride_a, stride_b, stride_c, bsz,
      cublasGemmAlgo_t(_config.gemm_algos[1]));

  // A need to transpose.
  cublasOperation_t op_a =
      (_config.op_A == CUBLAS_OP_T ? CUBLAS_OP_N : CUBLAS_OP_T);

  stride_a = _config.m * _config.k;
  stride_b = _config.m * _config.n;
  stride_c = _config.n * _config.k;

  // Calculate d_B.
  cublas_strided_batched_gemm(
      handle, _config.k, _config.n, _config.m, &_config.alpha, &_config.beta,
      _buffer_a, d_output, inpGradB, op_a, CUBLAS_OP_N, stride_a, stride_b,
      stride_c, bsz, cublasGemmAlgo_t(_config.gemm_algos[2]));
}

template <typename T>
inline void StridedBatchGemm<T>::SetConfig(int m, int n, int k) {
  _config.SetConfig(m, n, k);
}

template class StridedBatchGemm<float>;
template class StridedBatchGemm<__half>;
