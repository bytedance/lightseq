/* Copyright 2021 The LightSeq Team
*/
#pragma once

#include <assert.h>
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>

template <typename T, typename S>
int cublas_lt_matmul(cublasLtHandle_t handle, cublasLtMatmulDesc_t matmulDesc,
                     cublasLtMatrixLayout_t ADesc, cublasLtMatrixLayout_t BDesc,
                     cublasLtMatrixLayout_t CDesc, T *A, T *B, S *C, S *alpha,
                     S *beta, cudaStream_t stream);

template <typename T, typename S>
float test_lt_matmul(cublasLtHandle_t handle, int C, int B, int O, int H, T *X,
                     T *W, S *Y, S *alpha, S *beta, int iteration);