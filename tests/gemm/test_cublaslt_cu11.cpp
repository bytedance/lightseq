#include <iostream>
#include <sys/time.h>
#include <cuda_profiler_api.h>
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

int8_t float2int8(float f, float scale) {
  int8_t i = int8_t(f * scale);
  if (i < -127) i = -127;
  if (i > 127) i = 127;
  return i;
}

template <typename T, typename S>
void allocate_memory(int m, int n, int k, T **A, T **B, S **C) {
  cudaMallocManaged(A, m * k * sizeof(T));
  cudaMallocManaged(B, k * n * sizeof(T));
  cudaMallocManaged(C, m * n * sizeof(S));
}

template <typename T, typename S>
void free_memory(T *A, T *B, S *C) {
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}

template <typename T, typename S>
int cublas_lt_matmul(cublasLtHandle_t handle,
                     cublasLtMatmulDesc_t operationDesc,
                     cublasLtMatrixLayout_t ADesc, cublasLtMatrixLayout_t BDesc,
                     cublasLtMatrixLayout_t CDesc, T *A, T *B, S *C, S *alpha,
                     S *beta) {
  cublasStatus_t status;
  status = cublasLtMatmul(handle, operationDesc, alpha, A, ADesc, B, BDesc,
                          beta, C, CDesc, C, CDesc, nullptr, nullptr, 0, 0);

  if (status == CUBLAS_STATUS_SUCCESS)
    return 1;
  else
    return -1;
}

template <typename T, typename S>
void test_matmul(cublasLtHandle_t handle, int m, int n, int k, T *A, T *B, S *C,
                 S *alpha, S *beta, int iteration) {
  cublasLtMatmulDesc_t operationDesc;
  cublasLtMatrixLayout_t ADesc, BDesc, CDesc;
  cudaDataType_t AType, BType, CType, scaleType;
  cublasComputeType_t ComputeType;
  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;

  if (std::is_same<T, float>::value) {
    AType = BType = CType = scaleType = CUDA_R_32F;
    ComputeType = CUBLAS_COMPUTE_32F;
  } else if (std::is_same<T, __half>::value) {
    AType = BType = CType = scaleType = CUDA_R_16F;
    ComputeType = CUBLAS_COMPUTE_16F;
  } else if (std::is_same<T, int8_t>::value) {
    AType = BType = CUDA_R_8I;
    CType = scaleType = CUDA_R_32I;
    ComputeType = CUBLAS_COMPUTE_32I;
  } else {
    printf("Not supported data type.");
    return;
  }

  cublasLtMatrixLayoutCreate(&ADesc, AType, k, m, k);
  cublasLtMatrixLayoutCreate(&BDesc, BType, n, k, n);
  cublasLtMatrixLayoutCreate(&CDesc, CType, n, m, n);
  cublasLtMatmulDescCreate(&operationDesc, ComputeType, scaleType);

  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                 &transa, sizeof(cublasOperation_t));
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                 &transb, sizeof(cublasOperation_t));

  float total_time = 0;
  for (int i = 0; i < iteration; ++i) {
    struct timeval start, end;
    cudaDeviceSynchronize();
    cudaProfilerStart();
    gettimeofday(&start, NULL);
    int success = cublas_lt_matmul(handle, operationDesc, BDesc, ADesc, CDesc,
                                   B, A, C, alpha, beta);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    cudaProfilerStop();
    if (success > 0 && i > 0)
      total_time += (end.tv_sec - start.tv_sec) * 1000 +
                    (end.tv_usec - start.tv_usec) * 0.001;
  }
  if (total_time > 0) printf("%.3f ms\n", total_time / (iteration - 1));
}

int main() {
  int m = 8192, n = 1024, k = 4096;
  printf("shape: (%d, %d) x (%d, %d)\n", m, k, k, n);
  int iteration = 10;

  float *fA, *fB, *fC;
  __half *hA, *hB, *hC;
  int8_t *iA, *iB;
  int32_t *iC;
  float f_alpha = 1, f_beta = 0;
  __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
  int32_t i_alpha = 1, i_beta = 0;
  allocate_memory(m, n, k, &fA, &fB, &fC);
  allocate_memory(m, n, k, &hA, &hB, &hC);
  allocate_memory(m, n, k, &iA, &iB, &iC);
  for (int i = 0; i < m * k; ++i) {
    fA[i] = float(i % 255 - 127) / 127;
    hA[i] = __float2half_rn(fA[i]);
    iA[i] = float2int8(fA[i], 127);
  }
  for (int i = 0; i < k * n; ++i) {
    fB[i] = float(i % 255 - 127) / 127;
    hB[i] = __float2half_rn(fB[i]);
    iB[i] = float2int8(fB[i], 127);
  }

  cublasLtHandle_t handle;
  cublasLtCreate(&handle);

  printf(">>>>>>>>>>>>>>>>> test fp32 >>>>>>>>>>>>>>>>>\n");
  test_matmul(handle, m, n, k, fA, fB, fC, &f_alpha, &f_beta, iteration);

  printf(">>>>>>>>>>>>>>>>> test fp16 >>>>>>>>>>>>>>>>>\n");
  test_matmul(handle, m, n, k, hA, hB, hC, &h_alpha, &h_beta, iteration);

  printf(">>>>>>>>>>>>>>>>> test int8 >>>>>>>>>>>>>>>>>\n");
  test_matmul(handle, m, n, k, iA, iB, iC, &i_alpha, &i_beta, iteration);

  printf(">>>>>>>>>>>>>>>>> compare result >>>>>>>>>>>>>>>>>\n");
  printf("fp32: ");
  for (int i = 0; i < 10; ++i) printf("%.5f%c", fC[i], " \n"[i == 9]);
  printf("fp16: ");
  for (int i = 0; i < 10; ++i) printf("%.5f%c", float(hC[i]), " \n"[i == 9]);
  printf("int8: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", float(iC[i]) / 127 / 127, " \n"[i == 9]);

  free_memory(iA, iB, iC);
  free_memory(fA, fB, fC);
  free_memory(hA, hB, hC);
  return 0;
}
