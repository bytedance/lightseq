#include <sys/time.h>
#include <cuda_profiler_api.h>
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <algorithm>
#include <vector>

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
      C[i * n + j] = 0;
      for (int kk = 0; kk < k; ++kk) {
        C[i * n + j] += A[i * k + kk] * B[kk * n + j];
      }
    }
  }
}

template <typename T, typename S>
void allocate_memory(int B, int O, int H, T **X, T **W, S **Y, T **X_T,
                     T **W_T) {
  cudaMallocManaged(X, B * H * sizeof(T));
  cudaMallocManaged(W, O * H * sizeof(T));
  cudaMallocManaged(Y, B * O * sizeof(S));
  cudaMallocManaged(X_T, H * B * sizeof(T));
  cudaMallocManaged(W_T, H * O * sizeof(T));
}

template <typename T, typename S>
void free_memory(T *X, T *W, S *Y, T *X_T, T *W_T) {
  cudaFree(X);
  cudaFree(W);
  cudaFree(Y);
  cudaFree(X_T);
  cudaFree(W_T);
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
  else {
    return -1;
  }
}

template <typename T, typename S>
void test_lt_matmul(cublasLtHandle_t handle, int B, int O, int H, T *X, T *W,
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
  cublasOperation_t opNTrans = CUBLAS_OP_N;
  cublasOperation_t opTrans = CUBLAS_OP_T;

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
                                 &opNTrans, sizeof(cublasOperation_t));
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                 &opNTrans, sizeof(cublasOperation_t));

  float total_time = 0;
  for (int i = 0; i < iteration; ++i) {
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    int success = cublas_lt_matmul(handle, operationDesc, WDesc, XDesc, YDesc,
                                   W, X, Y, alpha, beta);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (success > 0 && i > 0) total_time += time;
  }
  if (total_time > 0) printf("%.3f ms\n", total_time / (iteration - 1));
  cublasLtMatmulDescDestroy(operationDesc);
  cublasLtMatrixLayoutDestroy(XDesc);
  cublasLtMatrixLayoutDestroy(WDesc);
  cublasLtMatrixLayoutDestroy(YDesc);
}

void test_lt_matmul_int8(cublasLtHandle_t handle, int B, int O, int H,
                         int8_t *X, int8_t *W, int32_t *Y, int32_t *alpha,
                         int32_t *beta, int iteration) {
  cublasLtMatmulDesc_t operationDesc;
  cublasLtMatrixLayout_t XDesc, WDesc, YDesc;
  cudaDataType_t XType, WType, YType;
#if CUBLAS_VER_MAJOR == 11
  cublasComputeType_t ComputeType;
  cudaDataType_t scaleType;
#else
  cudaDataType_t ComputeType;
#endif
  cublasOperation_t opNTrans = CUBLAS_OP_N;
  cublasOperation_t opTrans = CUBLAS_OP_T;
  cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;
  cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;

  cublasLtMatrixTransformDesc_t transformDesc;
  int8_t *Xtransform, *Wtransform;
  int32_t *Ytransform;
  cublasLtMatrixLayout_t XtransformDesc, WtransformDesc, YtransformDesc;
  float transformAlpha = 1.0f, transformBeta = 0.0f;

  XType = WType = CUDA_R_8I;
  YType = CUDA_R_32I;
#if CUBLAS_VER_MAJOR == 11
  ComputeType = CUBLAS_COMPUTE_32I;
  scaleType = CUDA_R_32I;
#else
  ComputeType = CUDA_R_32I;
#endif

  int ldXtransform = 32 * B;
  int ldWtransform = 32 * O;
  int ldYtransform = 32 * B;

  cudaMalloc(reinterpret_cast<void **>(&Xtransform), sizeof(int8_t) * B * H);
  cudaMalloc(reinterpret_cast<void **>(&Wtransform), sizeof(int8_t) * O * H);
  cudaMalloc(reinterpret_cast<void **>(&Ytransform), sizeof(int32_t) * B * O);

  cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_32F);

  cublasLtMatrixLayoutCreate(&XDesc, XType, H, B, H);
  cublasLtMatrixLayoutCreate(&WDesc, WType, H, O, H);
  cublasLtMatrixLayoutCreate(&YDesc, YType, O, B, O);
#if CUBLAS_VER_MAJOR == 11
  cublasLtMatmulDescCreate(&operationDesc, ComputeType, scaleType);
#else
  cublasLtMatmulDescCreate(&operationDesc, ComputeType);
#endif
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                 &opNTrans, sizeof(cublasOperation_t));
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                 &opTrans, sizeof(cublasOperation_t));

  cublasLtMatrixLayoutCreate(&XtransformDesc, CUDA_R_8I, B, H, ldXtransform);
  cublasLtMatrixLayoutSetAttribute(XtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_COL32, sizeof(order_COL32));
  cublasLtMatrixLayoutCreate(&WtransformDesc, CUDA_R_8I, O, H, ldWtransform);
  cublasLtMatrixLayoutSetAttribute(WtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_COL4_4R2_8C,
                                   sizeof(order_COL4_4R2_8C));
  cublasLtMatrixLayoutCreate(&YtransformDesc, CUDA_R_32I, B, O, ldYtransform);
  cublasLtMatrixLayoutSetAttribute(YtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_COL32, sizeof(order_COL32));

  cublasLtMatrixTransformDescSetAttribute(transformDesc,
                                          CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA,
                                          &opTrans, sizeof(cublasOperation_t));
  cublasLtMatrixTransform(handle, transformDesc, &transformAlpha, X, XDesc,
                          &transformBeta, NULL, NULL, Xtransform,
                          XtransformDesc, 0);
  cublasLtMatrixTransform(handle, transformDesc, &transformAlpha, W, WDesc,
                          &transformBeta, NULL, NULL, Wtransform,
                          WtransformDesc, 0);

  float total_time = 0;
  for (int i = 0; i < iteration; ++i) {
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    int success = cublas_lt_matmul(handle, operationDesc, XtransformDesc,
                                   WtransformDesc, YtransformDesc, Xtransform,
                                   Wtransform, Ytransform, alpha, beta);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (success > 0 && i > 0) total_time += time;
  }
  cublasLtMatrixTransformDescSetAttribute(transformDesc,
                                          CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA,
                                          &opTrans, sizeof(cublasOperation_t));
  cublasLtMatrixTransform(handle, transformDesc, &transformAlpha, Ytransform,
                          YtransformDesc, &transformBeta, NULL, NULL, Y, YDesc,
                          0);
  cudaDeviceSynchronize();
  cudaFree(Xtransform);
  cudaFree(Wtransform);
  cudaFree(Ytransform);
  if (total_time > 0) printf("%.3f ms\n", total_time / (iteration - 1));
}

void _main(int B, int O, int H, int iteration, bool debug) {
  printf(
      ">>>>>>>>>>>>>>>>>>>> shape: X(%d, %d), W(%d, %d) >>>>>>>>>>>>>>>>>>>>\n",
      B, H, O, H);

  float *Y;
  if (debug) cudaMallocManaged(&Y, B * O * sizeof(float));

  float *fX, *fW, *fY, *fX_T, *fW_T;
  __half *hX, *hW, *hY, *hX_T, *hW_T;
  int8_t *iX, *iW, *iX_T, *iW_T;
  int32_t *iY;
  allocate_memory(B, O, H, &fX, &fW, &fY, &fX_T, &fW_T);
  allocate_memory(B, O, H, &hX, &hW, &hY, &hX_T, &hW_T);
  allocate_memory(B, O, H, &iX, &iW, &iY, &iX_T, &iW_T);

  float f_alpha = 1, f_beta = 0;
  __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
  int32_t i_alpha = 1, i_beta = 0;

  for (int i = 0; i < B * H; ++i) {
    fX[i] = float(i % 255 - 127) / 127;
    hX[i] = __float2half_rn(fX[i]);
    iX[i] = float2int8(fX[i], 127);
  }
  for (int i = 0; i < O * H; ++i) {
    fW[i] = float(i % 255 - 127) / 127;
    hW[i] = __float2half_rn(fW[i]);
    iW[i] = float2int8(fW[i], 127);
  }

  transpose(fX, fX_T, B, H);
  transpose(fW, fW_T, O, H);
  transpose(hX, hX_T, B, H);
  transpose(hW, hW_T, O, H);
  transpose(iX, iX_T, B, H);
  transpose(iW, iW_T, O, H);
  if (debug) matmul(fX, fW_T, Y, B, O, H);

  cublasLtHandle_t handle;
  cublasLtCreate(&handle);

  printf(">>>>> test fp32 >>>>>\n");
  test_lt_matmul(handle, B, O, H, fX, fW_T, fY, &f_alpha, &f_beta, iteration);

  printf(">>>>> test fp16 >>>>>\n");
  test_lt_matmul(handle, B, O, H, hX, hW_T, hY, &h_alpha, &h_beta, iteration);

  printf(">>>>> test int8 >>>>>\n");
  test_lt_matmul_int8(handle, B, O, H, iX, iW, iY, &i_alpha, &i_beta,
                      iteration);

  float fe = 0, he = 0, ie = 0;
  printf(">>>>> compare result >>>>>\n");
  if (debug) {
    printf("oracle:\n  ");
    for (int i = 0; i < 10; ++i)
      printf("%.5f%c", Y[B * O - 1 - i], " \n"[i == 9]);
  }

  printf("fp32:\n  ");
  for (int i = 0; i < 10; ++i) printf("%.5f%c", fY[i], " \n"[i == 9]);
  for (int i = 0; i < B * O; ++i) fe += fabs((debug ? Y[i] : fY[i]) - fY[i]);
  printf("  diff: %.5f\n", fe / B / O);

  printf("fp16:\n  ");
  for (int i = 0; i < 10; ++i) printf("%.5f%c", float(hY[i]), " \n"[i == 9]);
  for (int i = 0; i < B * O; ++i)
    he += fabs((debug ? Y[i] : fY[i]) - float(hY[i]));
  printf("  diff: %.5f\n", he / B / O);

  printf("int8:\n  ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", float(iY[i]) / 127 / 127, " \n"[i == 9]);
  for (int i = 0; i < B * O; ++i)
    ie += fabs((debug ? Y[i] : fY[i]) - float(iY[i]) / 127 / 127);
  printf("  diff: %.5f\n", ie / B / O);

  free_memory(fX, fW, fY, fX_T, fW_T);
  free_memory(hX, hW, hY, hX_T, hW_T);
  free_memory(iX, iW, iY, iX_T, iW_T);
  if (debug) cudaFree(Y);
}

int main() {
  int iteration = 50;
  bool debug = false;
  std::vector<int> Bs = {8, 16, 1024};
  std::vector<int> Os = {1024, 3072, 4096};
  std::vector<int> Hs = {1024, 4096};
  for (int i = 0; i < Bs.size(); ++i) {
    for (int j = 0; j < Os.size(); ++j) {
      for (int k = 0; k < Hs.size(); ++k) {
        _main(Bs[i], Os[j], Hs[k], iteration, debug);
      }
    }
  }
  return 0;
}
