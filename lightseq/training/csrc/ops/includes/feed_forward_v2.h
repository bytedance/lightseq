#pragma once

/* Copyright 2021 The LightSeq Team
   Copyright Microsoft DeepSpeed
   This file is adapted from Microsoft DeepSpeed
*/
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>

#include <array>

#include "cublaslt_wrappers.h"
#include "kernels.h"
#include "cuda_util.h"

template <typename T>
class FeedForwardV2 {
 public:
  struct Config {
    int bsz, m, n, k;
    T alpha, beta;
    int max_m, max_n, max_k;
    Config(int max_m, int max_n, int max_k, T alpha = 1.0, T beta = 0.0)
        : max_m(max_m), max_n(max_n), max_k(max_k), alpha(alpha), beta(beta) {}
    void SetConfig(int b, int mm, int nn, int kk) {
      bsz = b;
      m = mm;
      n = nn;
      k = kk;
    }
  };

  FeedForwardV2(Config config) : _config(config) {
    if (std::is_same<T, float>::value) {
      _AType = _BType = _CType = CUDA_R_32F;
#if CUBLAS_VER_MAJOR == 11
      _ComputeType = CUBLAS_COMPUTE_32F;
      _scaleType = CUDA_R_32F;
#else
      _ComputeType = CUDA_R_32F;
#endif
    } else if (std::is_same<T, __half>::value) {
      _AType = _BType = _CType = CUDA_R_16F;
#if CUBLAS_VER_MAJOR == 11
      _ComputeType = CUBLAS_COMPUTE_16F;
      _scaleType = CUDA_R_16F;
#else
      _ComputeType = CUDA_R_16F;
#endif
    }

#if CUBLAS_VER_MAJOR == 11
    cublasLtMatmulDescCreate(&_matmulDesc, _ComputeType, _scaleType);
#else
    cublasLtMatmulDescCreate(&_matmulDesc, _ComputeType);
#endif
  }

  ~FeedForwardV2() {
    if (_ADesc) cublasLtMatrixLayoutDestroy(_ADesc);
    if (_BDesc) cublasLtMatrixLayoutDestroy(_BDesc);
    if (_CDesc) cublasLtMatrixLayoutDestroy(_CDesc);
    if (_matmulDesc) cublasLtMatmulDescDestroy(_matmulDesc);
  }

  void Forward(const T *A, const T *B, T *C, int transA, int transB,
               cublasLtHandle_t handle, cudaStream_t stream) {
    if (transA)
      cublasLtMatmulDescSetAttribute(_matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                     &_opTrans, sizeof(_opTrans));
    if (transB)
      cublasLtMatmulDescSetAttribute(_matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                     &_opTrans, sizeof(_opTrans));

    int m = _config.m, n = _config.n, k = _config.k;
    cublasLtMatrixLayoutCreate(&_ADesc, _AType, transA ? k : m, transA ? m : k,
                               transA ? k : m);
    cublasLtMatrixLayoutCreate(&_BDesc, _BType, transB ? n : k, transB ? k : n,
                               transB ? n : k);
    cublasLtMatrixLayoutCreate(&_CDesc, _CType, m, n, m);

    if (_config.bsz > 1) {
      int64_t strideA = m * k, strideB = n * k, strideC = m * n;
      cublasLtMatrixLayoutSetAttribute(_ADesc,
                                       CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                       &_config.bsz, sizeof(_config.bsz));
      cublasLtMatrixLayoutSetAttribute(
          _ADesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA,
          sizeof(strideA));
      cublasLtMatrixLayoutSetAttribute(_BDesc,
                                       CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                       &_config.bsz, sizeof(_config.bsz));
      cublasLtMatrixLayoutSetAttribute(
          _BDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB,
          sizeof(strideB));
      cublasLtMatrixLayoutSetAttribute(_CDesc,
                                       CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                       &_config.bsz, sizeof(_config.bsz));
      cublasLtMatrixLayoutSetAttribute(
          _CDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC,
          sizeof(strideC));
    }

    cublas_lt_matmul(handle, _matmulDesc, _ADesc, _BDesc, _CDesc, A, B, C,
                     &_config.alpha, &_config.beta, stream);
  }

  inline void SetConfig(int bsz, int m, int n, int k) {
    _config.SetConfig(bsz, m, n, k);
  }

 private:
  Config _config;
  cudaDataType_t _AType, _BType, _CType;
#if CUBLAS_VER_MAJOR == 11
  cublasComputeType_t _ComputeType;
  cudaDataType_t _scaleType;
#else
  cudaDataType_t _ComputeType;
#endif
  cublasOperation_t _opTrans = CUBLAS_OP_T;
  cublasLtMatrixLayout_t _ADesc = NULL, _BDesc = NULL, _CDesc = NULL;
  cublasLtMatmulDesc_t _matmulDesc = NULL;
};
