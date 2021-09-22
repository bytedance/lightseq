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
    int max_bsz, max_m, max_n, max_k;
    bool transA, transB;
    Config(int max_bsz, int max_m, int max_n, int max_k, bool transA,
           bool transB, T alpha = 1.0, T beta = 0.0)
        : max_bsz(max_bsz),
          max_m(max_m),
          max_n(max_n),
          max_k(max_k),
          transA(transA),
          transB(transB),
          alpha(alpha),
          beta(beta) {}
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
    CHECK_GPU_ERROR(
        cublasLtMatmulDescCreate(&_matmulDesc, _ComputeType, _scaleType));
#else
    CHECK_GPU_ERROR(cublasLtMatmulDescCreate(&_matmulDesc, _ComputeType));
#endif
  }

  ~FeedForwardV2() {
    if (_ADesc) CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(_ADesc));
    if (_BDesc) CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(_BDesc));
    if (_CDesc) CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(_CDesc));
    if (_matmulDesc) CHECK_GPU_ERROR(cublasLtMatmulDescDestroy(_matmulDesc));
  }

  void Forward(const T *A, const T *B, T *C, cublasLtHandle_t handle,
               cudaStream_t stream) {
    if (_config.transA)
      CHECK_GPU_ERROR(cublasLtMatmulDescSetAttribute(
          _matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &_opTrans,
          sizeof(_opTrans)));
    if (_config.transB)
      CHECK_GPU_ERROR(cublasLtMatmulDescSetAttribute(
          _matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &_opTrans,
          sizeof(_opTrans)));

    int m = _config.m, n = _config.n, k = _config.k;
    CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(
        &_ADesc, _AType, _config.transA ? k : m, _config.transA ? m : k,
        _config.transA ? k : m));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(
        &_BDesc, _BType, _config.transB ? n : k, _config.transB ? k : n,
        _config.transB ? n : k));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&_CDesc, _CType, m, n, m));

    if (_config.bsz > 1) set_batch_size(_ADesc, _BDesc, _CDesc, m, n, k);

    cublas_lt_matmul(handle, _matmulDesc, _ADesc, _BDesc, _CDesc, A, B, C,
                     &_config.alpha, &_config.beta, stream);
  }

  void set_batch_size(cublasLtMatrixLayout_t &ADesc,
                      cublasLtMatrixLayout_t &BDesc,
                      cublasLtMatrixLayout_t &CDesc, int m, int n, int k) {
    int64_t strideA = m * k, strideB = n * k, strideC = m * n;
    CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(
        ADesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &_config.bsz,
        sizeof(_config.bsz)));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(
        ADesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA,
        sizeof(strideA)));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(
        BDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &_config.bsz,
        sizeof(_config.bsz)));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(
        BDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB,
        sizeof(strideB)));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(
        CDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &_config.bsz,
        sizeof(_config.bsz)));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(
        CDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC,
        sizeof(strideC)));
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
