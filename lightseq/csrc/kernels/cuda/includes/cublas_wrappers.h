/* Copyright 2021 The LightSeq Team
   Copyright Microsoft DeepSpeed
   This file is adapted from Microsoft DeepSpeed
*/
#pragma once

#include <assert.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include "cublas_algo_map.h"
namespace lightseq {
namespace cuda {
int cublas_gemm_ex(cublasHandle_t handle, cublasOperation_t transa,
                   cublasOperation_t transb, int m, int n, int k,
                   const float *alpha, const float *beta, const float *A,
                   const float *B, float *C,
                   cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT);

int cublas_gemm_ex(cublasHandle_t handle, cublasOperation_t transa,
                   cublasOperation_t transb, int m, int n, int k,
                   const float *alpha, const float *beta, const __half *A,
                   const __half *B, __half *C,
                   cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP);

int cublas_strided_batched_gemm(cublasHandle_t handle, int m, int n, int k,
                                const float *alpha, const float *beta,
                                const float *A, const float *B, float *C,
                                cublasOperation_t op_A, cublasOperation_t op_B,
                                int stride_A, int stride_B, int stride_C,
                                int batch,
                                cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT);

int cublas_strided_batched_gemm(
    cublasHandle_t handle, int m, int n, int k, const float *alpha,
    const float *beta, const __half *A, const __half *B, __half *C,
    cublasOperation_t op_A, cublasOperation_t op_B, int stride_A, int stride_B,
    int stride_C, int batch,
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP);

template <typename OutType, typename ScaleType>
void cublaslt_igemm(const int8_t *input_a, const int8_t *input_b,
                    OutType *output_c, int batch_count, int m, int n, int k,
                    int64_t stridea, int64_t strideb, int64_t stridec,
                    const ScaleType *alpha, const ScaleType *beta,
                    cublasLtHandle_t cublasLt_handle, cudaStream_t stream);

template <typename OutType, typename ScaleType>
void cublaslt_igemm(const int8_t *input_a, const int8_t *input_b,
                    OutType *output_c, int batch_count, int m, int n, int k,
                    int64_t stridea, int64_t strideb, int64_t stridec,
                    const ScaleType *alpha, const ScaleType *beta,
                    cublasLtHandle_t cublasLt_handle, cudaStream_t stream,
                    cublasLtMatmulAlgo_info &algo_info,
                    cublasAlgoMap &algo_map);

inline int round_up(int v, int d) { return (v + d - 1) / d * d; }

void cublasLtMM_withAlgo_i8IO(int8_t *res, int batchCount, int m, int n, int k,
                              int64_t stridea, int64_t strideb, int64_t stridec,
                              const float *alpha, const float *beta,
                              const int8_t *ATransform, const int8_t *kernel,
                              cublasLtHandle_t cublasLt_handle,
                              cudaStream_t stream, bool use_ORDER_COL32_2R_4R4);

void cublasLtMM_withAlgo_i8IO(int8_t *res, int batchCount, int m, int n, int k,
                              int64_t stridea, int64_t strideb, int64_t stridec,
                              const float *alpha, const float *beta,
                              const int8_t *ATransform, const int8_t *kernel,
                              cublasLtHandle_t cublasLt_handle,
                              cudaStream_t stream,
                              cublasLtMatmulAlgo_info &algo_info,
                              cublasAlgoMap &algo_map);
}  // namespace cuda
}  // namespace lightseq
