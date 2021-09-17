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
    int max_size_A, max_size_B;
    Config(int max_size_A, int max_size_B)
        : max_size_A(max_size_A),
          max_size_B(max_size_B) {}
  };

  FeedForwardV2(Config config) : _config(config) {
    if (config.max_size_A > 0) {
      _transA = true;
      // _transformA = cuda_malloc<T>(config.max_size_A);
    }
    if (config.max_size_B > 0) {
      _transB = true;
      // _transformB = cuda_malloc<T>(config.max_size_B);
    }

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
  }

  ~FeedForwardV2() {
    // if (_transA)
    //   cuda_free(_transformA);
    // if (_transB)
    //   cuda_free(_transformB);
    cublasLtMatrixLayoutDestroy(_ADesc);
    cublasLtMatrixLayoutDestroy(_BDesc);
    cublasLtMatrixLayoutDestroy(_CDesc);
    cublasLtMatmulDescDestroy(_matmulDesc);
  }

  void Forward(int bsz, int m, int n, int k, const T *A, const T *B, T *C,
               cublasLtHandle_t handle, cudaStream_t stream) {
    _strideA = m * k;
    _strideB = n * k;
    _strideC = m * n;
    
    if (_transA) {
      cublasLtMatrixLayoutCreate(&_ADesc, _AType, k, m, k);
      // cublasLtMatrixLayoutCreate(&_transformADesc, _AType, m, k, m);
    } else {
      cublasLtMatrixLayoutCreate(&_ADesc, _AType, m, k, m);
    }
    if (_transB) {
      cublasLtMatrixLayoutCreate(&_BDesc, _BType, n, k, n);
      // cublasLtMatrixLayoutCreate(&_transformBDesc, _BType, k, n, k);
    } else {
      cublasLtMatrixLayoutCreate(&_BDesc, _BType, k, n, k);
    }
    cublasLtMatrixLayoutCreate(&_CDesc, _CType, m, n, m);
      
    if (bsz > 1) {
      cublasLtMatrixLayoutSetAttribute(
          _ADesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bsz, sizeof(bsz));
      cublasLtMatrixLayoutSetAttribute(
          _ADesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &_strideA,
          sizeof(_strideA));
      cublasLtMatrixLayoutSetAttribute(
          _BDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bsz, sizeof(bsz));
      cublasLtMatrixLayoutSetAttribute(
          _BDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &_strideB,
          sizeof(_strideB));
      cublasLtMatrixLayoutSetAttribute(
          _CDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bsz, sizeof(bsz));
      cublasLtMatrixLayoutSetAttribute(
          _CDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &_strideC,
          sizeof(_strideC));
      // if (_transA) {
      //   cublasLtMatrixLayoutSetAttribute(
      //     _transformADesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bsz, sizeof(bsz));
      //   cublasLtMatrixLayoutSetAttribute(
      //     _transformADesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA,
      //     sizeof(strideA));
      // }
      // if (_transB) {
      //   cublasLtMatrixLayoutSetAttribute(
      //     _transformBDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bsz, sizeof(bsz));
      //   cublasLtMatrixLayoutSetAttribute(
      //     _transformBDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB,
      //     sizeof(strideB));
      // }
    }

#if CUBLAS_VER_MAJOR == 11
    cublasLtMatmulDescCreate(&_matmulDesc, _ComputeType, _scaleType);
#else
    cublasLtMatmulDescCreate(&_matmulDesc, _ComputeType);
#endif
    if (_transA) {
      cublasLtMatmulDescSetAttribute(
        _matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &_opTrans, sizeof(_opTrans));
    }
    if (_transB) {
      cublasLtMatmulDescSetAttribute(
        _matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &_opTrans, sizeof(_opTrans));
    }

    cublas_lt_matmul(handle, _matmulDesc, _ADesc, _BDesc, _CDesc, A, B, C, &_alpha, &_beta, stream);
  }
 
 private:
  Config _config;
  // bool _transA, _transB;
  // T *_transformA, *_transformB;
  T _alpha = T(1.), _beta = T(0.);
  int64_t _strideA, _strideB, _strideC;
  cudaDataType_t _AType, _BType, _CType;
#if CUBLAS_VER_MAJOR == 11
  cublasComputeType_t _ComputeType;
  cudaDataType_t _scaleType;
#else
  cudaDataType_t _ComputeType;
#endif
  cublasOperation_t _opTrans = CUBLAS_OP_T;
  cublasLtMatrixLayout_t _ADesc, _BDesc, _CDesc;
  cublasLtMatmulDesc_t _matmulDesc;
  // cublasLtMatrixLayout_t _transformADesc, _transformBDesc;
  // cublasLtMatrixTransformDesc_t _transformDesc;
};
