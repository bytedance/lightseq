/* Copyright 2021 The LightSeq Team
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

int cublas_lt_matmul(cublasLtHandle_t handle, cublasLtMatmulDesc_t matmulDesc,
                     cublasLtMatrixLayout_t ADesc, cublasLtMatrixLayout_t BDesc,
                     cublasLtMatrixLayout_t CDesc, const float *A,
                     const float *B, float *C, float *alpha, float *beta,
                     cudaStream_t stream);

int cublas_lt_matmul(cublasLtHandle_t handle, cublasLtMatmulDesc_t matmulDesc,
                     cublasLtMatrixLayout_t ADesc, cublasLtMatrixLayout_t BDesc,
                     cublasLtMatrixLayout_t CDesc, const __half *A,
                     const __half *B, __half *C, __half *alpha, __half *beta,
                     cudaStream_t stream);
