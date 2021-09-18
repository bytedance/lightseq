/* Copyright 2021 The LightSeq Team
 */
#include "cublaslt_wrappers.h"

int cublas_lt_matmul(cublasLtHandle_t handle, cublasLtMatmulDesc_t matmulDesc,
                     cublasLtMatrixLayout_t ADesc, cublasLtMatrixLayout_t BDesc,
                     cublasLtMatrixLayout_t CDesc, const float *A,
                     const float *B, float *C, float *alpha, float *beta,
                     cudaStream_t stream) {
  cublasStatus_t status;
  status = cublasLtMatmul(handle, matmulDesc, alpha, A, ADesc, B, BDesc, beta,
                          C, CDesc, C, CDesc, nullptr, nullptr, 0, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! kernel execution error. (error: %d) \n", (int)status);
    return EXIT_FAILURE;
  }
  return 0;
}

int cublas_lt_matmul(cublasLtHandle_t handle, cublasLtMatmulDesc_t matmulDesc,
                     cublasLtMatrixLayout_t ADesc, cublasLtMatrixLayout_t BDesc,
                     cublasLtMatrixLayout_t CDesc, const __half *A,
                     const __half *B, __half *C, __half *alpha, __half *beta,
                     cudaStream_t stream) {
  cublasStatus_t status;
  status = cublasLtMatmul(handle, matmulDesc, alpha, A, ADesc, B, BDesc, beta,
                          C, CDesc, C, CDesc, nullptr, nullptr, 0, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! kernel execution error. (error: %d) \n", (int)status);
    return EXIT_FAILURE;
  }
  return 0;
}
