/* Copyright 2021 The LightSeq Team
*/
#include "cublaslt_wrappers.h"


template <typename T>
int cublas_lt_matmul(cublasLtHandle_t handle, cublasLtMatmulDesc_t matmulDesc,
                     cublasLtMatrixLayout_t ADesc, cublasLtMatrixLayout_t BDesc,
                     cublasLtMatrixLayout_t CDesc, T *A, T *B, T *C, T *alpha,
                     T *beta, cudaStream_t stream) {
  cublasStatus_t status;
  status = cublasLtMatmul(handle, matmulDesc, alpha, A, ADesc, B, BDesc, beta,
                          C, CDesc, C, CDesc, nullptr, nullptr, 0, stream);
  return status;
}
