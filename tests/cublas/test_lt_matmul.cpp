#include <sys/time.h>
#include <cuda_profiler_api.h>
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <algorithm>

int8_t float2int8(float f, float scale) {
  int8_t i = int8_t(f * scale);
  if (i < -127) i = -127;
  if (i > 127) i = 127;
  return i;
}

template <typename T>
void transpose(T *A, T *A_T, int m, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      A_T[j * m + i] = A[i * n + j];
    }
  }
}

void matmul(float *A, float *B, float *C, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      C[i*n+j] = 0;
      for (int kk = 0; kk < k; ++kk) {
        C[i*n+j] += A[i*k+kk] * B[kk*n+j];
      }
    }
  }
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
                     cublasLtMatrixLayout_t XDesc, cublasLtMatrixLayout_t WDesc,
                     cublasLtMatrixLayout_t YDesc, T *A, T *B, S *C, S *alpha,
                     S *beta) {
  cublasStatus_t status;
  status = cublasLtMatmul(handle, operationDesc, alpha, A, XDesc, B, WDesc,
                          beta, C, YDesc, C, YDesc, nullptr, nullptr, 0, 0);

  if (status == CUBLAS_STATUS_SUCCESS)
    return 1;
  else
    return -1;
}

template <typename T, typename S>
void test_lt_matmul(cublasLtHandle_t handle, int B, int H, int O, T *X, T *W,
                    S *Y, S *alpha, S *beta, int iteration) {
  cublasLtMatmulDesc_t operationDesc;
  cublasLtMatrixLayout_t XDesc, WDesc, YDesc;
  cudaDataType_t XType, WType, YType;
#if CUBLAS_VER_MAJOR == 11
  cublasComputeType_t ComputeType;
  cudaDataType_t scaleType;
#else
  cudaDataType_t ComputeType;
#endif
  cublasOperation_t transX = CUBLAS_OP_N;
  cublasOperation_t transW = CUBLAS_OP_N;

  if (std::is_same<T, float>::value) {
    XType = WType = YType = CUDA_R_32F;
#if CUBLAS_VER_MAJOR == 11
    ComputeType = CUBLAS_COMPUTE_32F;
    scaleType = CUDA_R_32F;
#else
    ComputeType = CUDA_R_32F;
#endif
  } else if (std::is_same<T, __half>::value) {
    XType = WType = YType = CUDA_R_16F;
#if CUBLAS_VER_MAJOR == 11
    ComputeType = CUBLAS_COMPUTE_16F;
    scaleType = CUDA_R_16F;
#else
    ComputeType = CUDA_R_16F;
#endif
  } else if (std::is_same<T, int8_t>::value) {
    XType = WType = CUDA_R_8I;
    YType = CUDA_R_32I;
#if CUBLAS_VER_MAJOR == 11
    ComputeType = CUBLAS_COMPUTE_32I;
    scaleType = CUDA_R_32I;
#else
    ComputeType = CUDA_R_32I;
#endif
  } else {
    printf("Not supported data type.");
    return;
  }

  cublasLtMatrixLayoutCreate(&XDesc, XType, H, B, H);
  cublasLtMatrixLayoutCreate(&WDesc, WType, O, H, O);
  cublasLtMatrixLayoutCreate(&YDesc, YType, O, B, O);
#if CUBLAS_VER_MAJOR == 11
  cublasLtMatmulDescCreate(&operationDesc, ComputeType, scaleType);
#else
  cublasLtMatmulDescCreate(&operationDesc, ComputeType);
#endif
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                 &transW, sizeof(cublasOperation_t));
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                 &transX, sizeof(cublasOperation_t));

  float total_time = 0;
  for (int i = 0; i < iteration; ++i) {
    struct timeval start, end;
    cudaDeviceSynchronize();
    cudaProfilerStart();
    gettimeofday(&start, NULL);
    int success = cublas_lt_matmul(handle, operationDesc, WDesc, XDesc, YDesc,
                                   W, X, Y, alpha, beta);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    cudaProfilerStop();
    if (success > 0 && i > 0)
      total_time += (end.tv_sec - start.tv_sec) * 1000 +
                    (end.tv_usec - start.tv_usec) * 0.001;
  }
  if (total_time > 0) printf("%.3f ms\n", total_time / (iteration - 1));
  cublasLtMatmulDescDestroy(operationDesc);
  cublasLtMatrixLayoutDestroy(XDesc);
  cublasLtMatrixLayoutDestroy(WDesc);
  cublasLtMatrixLayoutDestroy(YDesc);
}

void test_lt_matmul_int8(cublasLtHandle_t handle, int B, int H, int O, int8_t *X, int8_t *W,
                    int32_t *Y, int32_t *alpha, int32_t *beta, int iteration) {
  cublasLtMatmulDesc_t operationDesc;
  cublasLtMatrixLayout_t XDesc, WDesc, YDesc;
  cudaDataType_t XType, WType, YType;
#if CUBLAS_VER_MAJOR == 11
  cublasComputeType_t ComputeType;
  cudaDataType_t scaleType;
#else
  cudaDataType_t ComputeType;
#endif
  cublasOperation_t transX = CUBLAS_OP_T;
  cublasOperation_t transW = CUBLAS_OP_N;
  cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;
  cublasLtOrder_t order_B = CUBLASLT_ORDER_COL4_4R2_8C;

  XType = WType = CUDA_R_8I;
  YType = CUDA_R_32I;
#if CUBLAS_VER_MAJOR == 11
  ComputeType = CUBLAS_COMPUTE_32I;
  scaleType = CUDA_R_32I;
#else
  ComputeType = CUDA_R_32I;
#endif

  int ldX = 32 * B;
  int ldW = 32 * O;
  int ldY = 32 * O;

  cublasLtMatrixLayoutCreate(&XDesc, XType, B, H, ldX);
  cublasLtMatrixLayoutSetAttribute(XDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_B, sizeof(cublasLtOrder_t));
  cublasLtMatrixLayoutCreate(&WDesc, WType, O, H, ldW);
  cublasLtMatrixLayoutSetAttribute(WDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(cublasLtOrder_t));
  cublasLtMatrixLayoutCreate(&YDesc, YType, O, B, ldY);
  cublasLtMatrixLayoutSetAttribute(YDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(cublasLtOrder_t));
#if CUBLAS_VER_MAJOR == 11
  cublasLtMatmulDescCreate(&operationDesc, ComputeType, scaleType);
#else
  cublasLtMatmulDescCreate(&operationDesc, ComputeType);
#endif
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                 &transW, sizeof(cublasOperation_t));
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                 &transX, sizeof(cublasOperation_t));

  float total_time = 0;
  for (int i = 0; i < iteration; ++i) {
    struct timeval start, end;
    cudaDeviceSynchronize();
    cudaProfilerStart();
    gettimeofday(&start, NULL);
    int success = cublas_lt_matmul(handle, operationDesc, WDesc, XDesc, YDesc,
                                   W, X, Y, alpha, beta);
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
  // Y = X * W^T
  // Y^T = W * X^T
  int B = 32, H = 64, O = 128;
  printf("shape: X(%d, %d), W(%d, %d)\n", B, H, O, H);
  int iteration = 10;

  float *Y;
  float *fX, *fW, *fY, *fW_T;
  __half *hX, *hW, *hY, *hW_T;
  int8_t *iX, *iW, *iX_T, *iW_T; int32_t *iY;
  float f_alpha = 1, f_beta = 0;
  __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
  int32_t i_alpha = 1, i_beta = 0;
  cudaMallocManaged(&Y, B * O * sizeof(float));
  cudaMallocManaged(&fW_T, H * O * sizeof(float));
  cudaMallocManaged(&hW_T, H * O * sizeof(__half));
  cudaMallocManaged(&iX_T, H * B * sizeof(int8_t));
  cudaMallocManaged(&iW_T, H * O * sizeof(int8_t));
  allocate_memory(B, H, O, &fX, &fW, &fY);
  allocate_memory(B, H, O, &hX, &hW, &hY);
  allocate_memory(B, H, O, &iX, &iW, &iY);
  for (int i = 0; i < B * H; ++i) {
    fX[i] = float(i % 255 - 127) / 127;
    hX[i] = __float2half_rn(fX[i]);
    // iX[i] = float2int8(fX[i], 127);
    iX[i] = int8_t(1);
  }
  for (int i = 0; i < O * H; ++i) {
    fW[i] = float(i % 255 - 127) / 127;
    hW[i] = __float2half_rn(fW[i]);
    // iW[i] = float2int8(fW[i], 127);
    iW[i] = int8_t(1);
  }
  transpose(fW, fW_T, O, H);
  transpose(hW, hW_T, O, H);
  transpose(iW, iW_T, O, H);
  // transpose(iX, iX_T, B, H);
  matmul(fX, fW_T, Y, B, O, H);

  cublasLtHandle_t handle;
  cublasLtCreate(&handle);

  // printf(">>>>>>>>>>>>>>>>> test fp32 >>>>>>>>>>>>>>>>>\n");
  // test_lt_matmul(handle, B, H, O, fX, fW_T, fY, &f_alpha, &f_beta, iteration);

  printf(">>>>>>>>>>>>>>>>> test fp16 >>>>>>>>>>>>>>>>>\n");
  // test_lt_matmul(handle, B, H, O, hX, hW_T, hY, &h_alpha, &h_beta, iteration);

  printf(">>>>>>>>>>>>>>>>> test int8 >>>>>>>>>>>>>>>>>\n");
  // test_lt_matmul(handle, B, H, O, iX, iW_T, iY, &i_alpha, &i_beta, iteration);
  test_lt_matmul_int8(handle, B, H, O, iX_T, iW_T, iY, &i_alpha, &i_beta, iteration);

  float fe = 0, he = 0, ie = 0; 
  printf(">>>>>>>>>>>>>>>>> compare result >>>>>>>>>>>>>>>>>\n");
  printf("oracle:\n  ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", Y[i], " \n"[i == 9]);
  // printf("fp32:\n  ");
  // for (int i = 0; i < 10; ++i)
  //   printf("%.5f%c", fY[i], " \n"[i == 9]);
  // for (int i = 0; i < B * O; ++i)
  //   fe += fabs(Y[i] - fY[i]);
  // printf("  error: %.5f\n", fe / B / O);
  printf("fp16:\n  ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", float(hY[i]), " \n"[i == 9]);
  for (int i = 0; i < B * O; ++i)
    he += fabs(Y[i] - float(hY[i]));
  printf("  error: %.5f\n", he / B / O);
  printf("int8:\n  ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", float(iY[i]), " \n"[i == 9]);
  for (int i = 0; i < B * O; ++i)
    ie += fabs(Y[i] - float(iY[i]));
  printf("  error: %.5f\n", ie / B / O);

  free_memory(iX, iW, iY);
  free_memory(fX, fW, fY);
  free_memory(hX, hW, hY);
  cudaFree(Y);
  cudaFree(fW_T);
  cudaFree(hW_T);
  cudaFree(iW_T);
  cudaFree(iX_T);
  return 0;
}
