#pragma once

/* Copyright 2021 The LightSeq Team
   Copyright Microsoft DeepSpeed
   This file is adapted from Microsoft DeepSpeed
*/
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>

#include <array>

#include "cuda_util.h"
#include "cublaslt_wrappers.h"
#include "kernels.h"
#include "int8_kernels.h"

template <typename T>
class FeedForwardV3 {
 public:
  struct Config {
    int m, n, k;
    Config() {}
    void SetConfig(int mm, int nn, int kk) {
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
  }

  ~FeedForwardV3() {
    if (_ADesc) CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(_ADesc));
    if (_BDesc) CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(_BDesc));
    if (_CDesc) CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(_CDesc));
    if (_BTransformDesc)
      CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(_BTransformDesc));
    if (_CTransformDesc)
      CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(_CTransformDesc));
    if (_matmulDesc) CHECK_GPU_ERROR(cublasLtMatmulDescDestroy(_matmulDesc));
    if (_transformDesc)
      CHECK_GPU_ERROR(cublasLtMatrixTransformDescDestroy(_transformDesc));
  }

  void Forward(const int8_t *A, const T *B, T *C, int8_t *B_buffer,
               int32_t *C_buffer, cublasLtHandle_t handle,
               cudaStream_t stream) {
    int m = _config.m, n = _config.n, k = _config.k;
    n = (n + 3) / 4 * 4;
    int size_B = n * k, size_C = m * n;
    int8_t *BT_buffer = B_buffer + size_B;
    int32_t *CT_buffer = C_buffer + size_C;

    float scale_A = 127, scale_B = 127, clip_max_A = 0.5, clip_max_B = 16.0;
    launch_quantize_tensor(B, B_buffer, n * k, scale_B, clip_max_B, stream);

    CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&_ADesc, _AType, m, k, m));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&_BDesc, _BType, k, n, k));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&_CDesc, _CType, m, n, m));

    // int ldBTransform = 256 * ((n + 7) / 8), ldCTransform = 32 * m;
    int ldBTransform = n, ldCTransform = m;
    CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&_BTransformDesc, _BType, n, k,
                                               ldBTransform));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&_CTransformDesc, _CType, m, n,
                                               ldCTransform));
    CHECK_GPU_ERROR(
        cublasLtMatrixLayoutSetAttribute(_ADesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                         &_order_COL32, sizeof(_order_COL32)));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(
        _BTransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &_order_COL4_4R2_8C,
        sizeof(_order_COL4_4R2_8C)));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(
        _CTransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &_order_COL32,
        sizeof(_order_COL32)));

    CHECK_GPU_ERROR(cublasLtMatrixTransformDescSetAttribute(
        _transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &_opTrans,
        sizeof(_opTrans)));

    CHECK_GPU_ERROR(cublasLtMatrixTransform(
        handle, _transformDesc, &transform_alpha, B_buffer, _BDesc,
        &transform_beta, NULL, NULL, BT_buffer, _BTransformDesc, stream));

    cublas_lt_matmul(handle, _matmulDesc, _ADesc, _BTransformDesc,
                     _CTransformDesc, A, BT_buffer, C_buffer, &alpha, &beta,
                     stream);

    CHECK_GPU_ERROR(cublasLtMatrixTransformDescSetAttribute(
        _transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &_opNTrans,
        sizeof(_opNTrans)));
    CHECK_GPU_ERROR(cublasLtMatrixTransform(
        handle, _transformDesc, &transform_alpha, C_buffer, _CTransformDesc,
        &transform_beta, NULL, NULL, CT_buffer, _CDesc, 0));
    launch_dequantize_tensor(CT_buffer, C, m * _config.n, scale_A * scale_B,
                             clip_max_A * clip_max_B, stream);
  }

  inline void SetConfig(int m, int n, int k) { _config.SetConfig(m, n, k); }

 private:
  Config _config;
  int32_t alpha = 1, beta = 0;
  float transform_alpha = 1.0, transform_beta = 0.0;
  cudaDataType_t _AType = CUDA_R_8I, _BType = CUDA_R_8I, _CType = CUDA_R_32I;
#if CUBLAS_VER_MAJOR == 11
  cublasComputeType_t _ComputeType = CUBLAS_COMPUTE_32I;
  cudaDataType_t _scaleType = CUDA_R_32I;
#else
  cudaDataType_t _ComputeType = CUDA_R_32I;
#endif
  cublasOperation_t _opTrans = CUBLAS_OP_T, _opNTrans = CUBLAS_OP_N;
  //   cublasLtOrder_t _order_COL32 = CUBLASLT_ORDER_COL32;
  //   cublasLtOrder_t _order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;
  cublasLtOrder_t _order_COL32 = CUBLASLT_ORDER_COL;
  cublasLtOrder_t _order_COL4_4R2_8C = CUBLASLT_ORDER_COL;
  cublasLtMatrixLayout_t _ADesc = NULL, _BDesc = NULL, _CDesc = NULL;
  cublasLtMatrixLayout_t _BTransformDesc = NULL, _CTransformDesc = NULL;
  cublasLtMatmulDesc_t _matmulDesc = NULL;
  cublasLtMatrixTransformDesc_t _transformDesc = NULL;
};
