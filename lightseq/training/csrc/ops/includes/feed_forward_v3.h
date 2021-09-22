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
class FeedForwardV3 {
 public:
  struct Config {
    int bsz, m, n, k;
    int32_t alpha, beta;
    float transform_alpha = 1.0f, transform_beta = 0.0f;
    int max_bsz, max_m, max_n, max_k;
    bool transA, transB;
    Config(int max_bsz, int max_m, int max_n, int max_k, bool transA,
           bool transB, int32_t alpha = 1, int32_t beta = 0)
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

  FeedForwardV3(Config config) : _config(config) {
#if CUBLAS_VER_MAJOR == 11
    CHECK_GPU_ERROR(
        cublasLtMatmulDescCreate(&_matmulDesc, _ComputeType, _scaleType));
#else
    CHECK_GPU_ERROR(cublasLtMatmulDescCreate(&_matmulDesc, _ComputeType));
#endif
    CHECK_GPU_ERROR(cublasLtMatmulDescSetAttribute(
        _matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &_opTrans, sizeof(_opTrans)));

    CHECK_GPU_ERROR(
        cublasLtMatrixTransformDescCreate(&_transformDesc, CUDA_R_32F));
    CHECK_GPU_ERROR(cublasLtMatrixTransformDescSetAttribute(
        _transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &_opTrans,
        sizeof(_opTrans)));

    _ATransform =
        cuda_malloc<int8_t>(_config.max_bsz * _config.max_m * _config.max_k);
    _BTransform =
        cuda_malloc<int8_t>(_config.max_bsz * _config.max_n * _config.max_k);
    _CTransform =
        cuda_malloc<int32_t>(_config.max_bsz * _config.max_m * _config.max_n);
  }

  ~FeedForwardV3() {
    if (_ADesc) CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(_ADesc));
    if (_BDesc) CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(_BDesc));
    if (_CDesc) CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(_CDesc));
    if (_ATransformDesc)
      CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(_ATransformDesc));
    if (_BTransformDesc)
      CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(_BTransformDesc));
    if (_CTransformDesc)
      CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(_CTransformDesc));
    if (_matmulDesc) CHECK_GPU_ERROR(cublasLtMatmulDescDestroy(_matmulDesc));
    if (_transformDesc)
      CHECK_GPU_ERROR(cublasLtMatrixTransformDescDestroy(_transformDesc));
    cuda_free(_ATransform);
    cuda_free(_BTransform);
    cuda_free(_CTransform);
  }

  void Forward(const T *A, const T *B, T *C, cublasLtHandle_t handle,
               cudaStream_t stream) {
    int m = _config.m, n = _config.n, k = _config.k;
    CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(
        &_ADesc, _AType, _config.transA ? k : m, _config.transA ? m : k,
        _config.transA ? k : m));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(
        &_BDesc, _BType, _config.transB ? k : n, _config.transB ? n : k,
        _config.transB ? k : n));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&_CDesc, _CType, m, n, m));
    if (_config.bsz > 1) set_batch_size(_ADesc, _BDesc, _CDesc, m, n, k);

    int ldATransform = 32 * m, ldBTransform = 256 * ((n + 7) / 8),
        ldCTransform = 32 * m;
    CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&_ATransformDesc, _AType, m, k,
                                               ldATransform));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&_BTransformDesc, _BType, n, k,
                                               ldBTransform));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&_CTransformDesc, _CType, m, n,
                                               ldCTransform));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(
        _ATransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &_order_COL32,
        sizeof(_order_COL32)));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(
        _BTransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &_order_COL4_4R2_8C,
        sizeof(_order_COL4_4R2_8C)));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(
        _CTransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &_order_COL32,
        sizeof(_order_COL32)));
    if (_config.bsz > 1)
      set_batch_size(_ATransformDesc, _BTransformDesc, _CTransformDesc, m, n,
                     k);

    CHECK_GPU_ERROR(cublasLtMatrixTransform(
        handle, _transformDesc, &_config.transform_alpha, A, _ADesc,
        &_config.transform_beta, NULL, NULL, _ATransform, _ATransformDesc,
        stream));
    CHECK_GPU_ERROR(cublasLtMatrixTransform(
        handle, _transformDesc, &_config.transform_alpha, B, _BDesc,
        &_config.transform_beta, NULL, NULL, _BTransform, _BTransformDesc,
        stream));

    cublas_lt_matmul(handle, _matmulDesc, _ATransformDesc, _BTransformDesc,
                     _CTransformDesc, _ATransform, _BTransform, _CTransform,
                     &_config.alpha, &_config.beta, stream);

    CHECK_GPU_ERROR(cublasLtMatrixTransformDescSetAttribute(
        _transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &_opNTrans,
        sizeof(_opNTrans)));
    CHECK_GPU_ERROR(cublasLtMatrixTransform(
        handle, _transformDesc, &_config.transform_alpha, _CTransform,
        _CTransformDesc, &_config.transform_beta, NULL, NULL, C, _CDesc, 0));
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
  cudaDataType_t _AType = CUDA_R_8I, _BType = CUDA_R_8I, _CType = CUDA_R_32I;
#if CUBLAS_VER_MAJOR == 11
  cublasComputeType_t _ComputeType = CUBLAS_COMPUTE_32I;
  cudaDataType_t _scaleType = CUDA_R_32I;
#else
  cudaDataType_t _ComputeType = CUDA_R_32I;
#endif
  cublasOperation_t _opTrans = CUBLAS_OP_T, _opNTrans = CUBLAS_OP_N;
  cublasLtOrder_t _order_COL32 = CUBLASLT_ORDER_COL32;
  cublasLtOrder_t _order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;
  cublasLtMatrixLayout_t _ADesc = NULL, _BDesc = NULL, _CDesc = NULL;
  cublasLtMatrixLayout_t _ATransformDesc = NULL, _BTransformDesc = NULL,
                         _CTransformDesc = NULL;
  cublasLtMatmulDesc_t _matmulDesc = NULL;
  cublasLtMatrixTransformDesc_t _transformDesc = NULL;
  int8_t *_ATransform, *_BTransform;
  int32_t *_CTransform;
};
