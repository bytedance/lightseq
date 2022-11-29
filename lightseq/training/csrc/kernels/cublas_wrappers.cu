/* Copyright 2021 The LightSeq Team
   Copyright Microsoft DeepSpeed
   This file is adapted from Microsoft DeepSpeed
*/
#ifdef __HIPCC__
#include <rocblas.h>
#endif
#include "cublas_wrappers.h"

int cublas_gemm_ex(cublasHandle_t handle, cublasOperation_t transa,
                   cublasOperation_t transb, int m, int n, int k,
                   const float *alpha, const float *beta, const float *A,
                   const float *B, float *C,
#ifdef __HIPCC__
                   rocblas_gemm_algo algo) {
  rocblas_status status = rocblas_gemm_ex(
      handle, transa, transb, m, n, k, (const void *)alpha, (const void *)A,
      rocblas_datatype_f32_r, (transa == rocblas_operation_none) ? m : k,
      (const void *)B, rocblas_datatype_f32_r,
      (transb == rocblas_operation_none) ? k : n, (const void *)beta, C,
      rocblas_datatype_f32_r, m, C, rocblas_datatype_f32_r, m,
      rocblas_datatype_f32_r, algo, 0, 0);

#else
                   cublasGemmAlgo_t algo) {
  cublasStatus_t status =
      cublasGemmEx(handle, transa, transb, m, n, k, (const void *)alpha,
                   (const void *)A, CUDA_R_32F, (transa == CUBLAS_OP_N) ? m : k,
                   (const void *)B, CUDA_R_32F, (transb == CUBLAS_OP_N) ? k : n,
                   (const void *)beta, C, CUDA_R_32F, m, CUDA_R_32F, algo);

#endif
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr,
            "!!!! kernel execution error. (m: %d, n: %d, k: %d, error: %d) \n",
            m, n, k, (int)status);
    return EXIT_FAILURE;
  }
  return 0;
}

int cublas_gemm_ex(cublasHandle_t handle, cublasOperation_t transa,
                   cublasOperation_t transb, int m, int n, int k,
                   const float *alpha, const float *beta, const __half *A,
                   const __half *B, __half *C,
#ifdef __HIPCC__
                   rocblas_gemm_algo algo) {
  __half alpha_value = __float2half(*alpha);
  __half beta_value = __float2half(*beta);

  rocblas_status status = rocblas_gemm_ex(
      handle, transa, transb, m, n, k, &alpha_value, (const void *)A,
      rocblas_datatype_f16_r, (transa == rocblas_operation_none) ? m : k,
      (const void *)B, rocblas_datatype_f16_r,
      (transb == rocblas_operation_none) ? k : n, &beta_value, (void *)C,
      rocblas_datatype_f16_r, m, (void *)C, rocblas_datatype_f16_r, m,
      rocblas_datatype_f16_r, algo, 0, 0);
#else
                   cublasGemmAlgo_t algo) {
  cublasStatus_t status = cublasGemmEx(
      handle, transa, transb, m, n, k, (const void *)alpha, (const void *)A,
      CUDA_R_16F, (transa == CUBLAS_OP_N) ? m : k, (const void *)B, CUDA_R_16F,
      (transb == CUBLAS_OP_N) ? k : n, (const void *)beta, (void *)C,
      CUDA_R_16F, m, CUDA_R_32F, algo);
#endif

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr,
            "!!!! kernel execution error. (m: %d, n: %d, k: %d, error: %d) \n",
            m, n, k, (int)status);
    return EXIT_FAILURE;
  }
  return 0;
}

int cublas_strided_batched_gemm(cublasHandle_t handle, int m, int n, int k,
                                const float *alpha, const float *beta,
                                const float *A, const float *B, float *C,
                                cublasOperation_t op_A, cublasOperation_t op_B,
                                int stride_A, int stride_B, int stride_C,
                                int batch,
#ifdef __HIPCC__
                                rocblas_gemm_algo algo) {
  rocblas_status status = rocblas_gemm_strided_batched_ex(
      handle, op_A, op_B, m, n, k, alpha, A, rocblas_datatype_f32_r,
      (op_A == rocblas_operation_none) ? m : k, stride_A, B,
      rocblas_datatype_f32_r, (op_B == rocblas_operation_none) ? k : n,
      stride_B, beta, C, rocblas_datatype_f32_r, m, stride_C, C,
      rocblas_datatype_f32_r, m, stride_C, batch, rocblas_datatype_f32_r, algo,
      0, 0);
#else
                                cublasGemmAlgo_t algo) {
  cublasStatus_t status = cublasGemmStridedBatchedEx(
      handle, op_A, op_B, m, n, k, alpha, A, CUDA_R_32F,
      (op_A == CUBLAS_OP_N) ? m : k, stride_A, B, CUDA_R_32F,
      (op_B == CUBLAS_OP_N) ? k : n, stride_B, beta, C, CUDA_R_32F, m, stride_C,
      batch, CUDA_R_32F, algo);
#endif
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr,
            "!!!! kernel execution error. (batch: %d, m: %d, n: %d, k: %d, "
            "error: %d) \n",
            batch, m, n, k, (int)status);
    return EXIT_FAILURE;
  }
  return 0;
}

int cublas_strided_batched_gemm(cublasHandle_t handle, int m, int n, int k,
                                const float *alpha, const float *beta,
                                const __half *A, const __half *B, __half *C,
                                cublasOperation_t op_A, cublasOperation_t op_B,
                                int stride_A, int stride_B, int stride_C,
                                int batch,
#ifdef __HIPCC__
                                rocblas_gemm_algo algo) {
  __half alpha_value = __float2half(*alpha);
  __half beta_value = __float2half(*beta);
  rocblas_status status = rocblas_gemm_strided_batched_ex(
      handle, op_A, op_B, m, n, k, &alpha_value, A, rocblas_datatype_f16_r,
      (op_A == rocblas_operation_none) ? m : k, stride_A, B,
      rocblas_datatype_f16_r, (op_B == rocblas_operation_none) ? k : n,
      stride_B, &beta_value, C, rocblas_datatype_f16_r, m, stride_C, C,
      rocblas_datatype_f16_r, m, stride_C, batch, rocblas_datatype_f16_r, algo,
      0, 0);
#else
                                cublasGemmAlgo_t algo) {
  cublasStatus_t status = cublasGemmStridedBatchedEx(
      handle, op_A, op_B, m, n, k, alpha, A, CUDA_R_16F,
      (op_A == CUBLAS_OP_N) ? m : k, stride_A, B, CUDA_R_16F,
      (op_B == CUBLAS_OP_N) ? k : n, stride_B, beta, C, CUDA_R_16F, m, stride_C,
      batch, CUDA_R_32F, algo);
#endif
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr,
            "!!!! kernel execution error. (m: %d, n: %d, k: %d, error: %d) \n",
            m, n, k, (int)status);
    return EXIT_FAILURE;
  }

  return 0;
}
