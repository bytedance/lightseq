#include <sys/time.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
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
void allocate_memory(int B, int O, int H, T **X, T **W, S **Y) {
  cudaMallocManaged(X, B * H * sizeof(T));
  cudaMallocManaged(W, O * H * sizeof(T));
  cudaMallocManaged(Y, B * O * sizeof(S));
}

template <typename T, typename S>
void free_memory(T *A, T *B, S *C) {
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}

template <typename T, typename S>
int cublas_gemm_ex(cublasHandle_t handle, cublasOperation_t transA,
                   cublasOperation_t transB, int m, int n, int k, T *A, T *B,
                   S *C, int lda, int ldb, int ldc, S *alpha, S *beta,
                   cublasGemmAlgo_t algo) {
  cudaDataType_t AType, BType, CType, ComputeType;
  if (std::is_same<T, float>::value) {
    AType = BType = CType = ComputeType = CUDA_R_32F;
  } else if (std::is_same<T, __half>::value) {
    AType = BType = CType = ComputeType = CUDA_R_16F;
  } else if (std::is_same<T, int8_t>::value) {
    AType = BType = CUDA_R_8I;
    CType = ComputeType = CUDA_R_32I;
  } else {
    printf("Not supported data type.");
    return -1;
  }
  cublasStatus_t status;
  status = cublasGemmEx(handle, transA, transB, m, n, k, alpha, A, AType, lda,
                        B, BType, ldb, beta, C, CType, ldc, ComputeType, algo);

  if (status == CUBLAS_STATUS_SUCCESS)
    return 1;
  else
    return -1;
}

template <typename T, typename S>
void test_gemm_ex(cublasHandle_t handle, int B, int O, int H, T *X, T *W, S *Y,
                  S *alpha, S *beta, int algo, int iteration) {
  float total_time = 0;
  for (int i = 0; i < iteration; ++i) {
    struct timeval start, end;
    cudaDeviceSynchronize();
    cudaProfilerStart();
    gettimeofday(&start, NULL);
    int success =
        cublas_gemm_ex(handle, CUBLAS_OP_T, CUBLAS_OP_N, O, B, H, W, X, Y, H, H,
                       O, alpha, beta, static_cast<cublasGemmAlgo_t>(algo));
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    cudaProfilerStop();
    if (success > 0 && i > 0)
      total_time += (end.tv_sec - start.tv_sec) * 1000 +
                    (end.tv_usec - start.tv_usec) * 0.001;
  }
  if (total_time > 0)
    printf("algo %d: %.3f ms\n", algo, total_time / (iteration - 1));
}

int main() {
  // Y = X * W^T
  // Y^T = W * X^T
  int B = 512, H = 256, O = 1024;
  printf("shape: X(%d, %d), W(%d, %d)\n", B, H, O, H);
  int start_algo = CUBLAS_GEMM_DEFAULT;
  int end_algo = CUBLAS_GEMM_ALGO23;
  int start_algo_t_op = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  int end_algo_t_op = CUBLAS_GEMM_ALGO15_TENSOR_OP;
  int iteration = 10;

  float *Y;
  float *fX, *fW, *fY, *fW_T;
  __half *hX, *hW, *hY;
  int8_t *iX, *iW; int32_t *iY;
  float f_alpha = 1, f_beta = 0;
  __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
  int32_t i_alpha = 1, i_beta = 0;
  cudaMallocManaged(&fW_T, H * O * sizeof(float));
  cudaMallocManaged(&Y, B * O * sizeof(float));
  allocate_memory(B, O, H, &fX, &fW, &fY);
  allocate_memory(B, O, H, &hX, &hW, &hY);
  allocate_memory(B, O, H, &iX, &iW, &iY);
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
  transpose(fW, fW_T, O, H);
  matmul(fX, fW_T, Y, B, O, H);
  cublasHandle_t handle;
  cublasCreate(&handle);

  printf(">>>>>>>>>>>>>>>>> test fp32 >>>>>>>>>>>>>>>>>\n");
  test_gemm_ex(handle, B, O, H, fX, fW, fY, &f_alpha, &f_beta, -1, iteration);
  test_gemm_ex(handle, B, O, H, fX, fW, fY, &f_alpha, &f_beta, 99, iteration);

  printf(">>>>>>>>>>>>>>>>> test fp16 >>>>>>>>>>>>>>>>>\n");
  test_gemm_ex(handle, B, O, H, hX, hW, hY, &h_alpha, &h_beta, -1, iteration);
  test_gemm_ex(handle, B, O, H, hX, hW, hY, &h_alpha, &h_beta, 99, iteration);

  printf(">>>>>>>>>>>>>>>>> test int8 >>>>>>>>>>>>>>>>>\n");
  test_gemm_ex(handle, B, O, H, iX, iW, iY, &i_alpha, &i_beta, -1, iteration);
  test_gemm_ex(handle, B, O, H, iX, iW, iY, &i_alpha, &i_beta, 99, iteration);

  float fe = 0, he = 0, ie = 0; 
  printf(">>>>>>>>>>>>>>>>> compare result >>>>>>>>>>>>>>>>>\n");
  printf("oracle:\n  ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", Y[i], " \n"[i == 9]);
  printf("fp32:\n  ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", fY[i], " \n"[i == 9]);
  for (int i = 0; i < B * O; ++i)
    fe += fabs(Y[i] - fY[i]);
  printf("  error: %.5f\n", fe / B / O);
  printf("fp16:\n  ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", float(hY[i]), " \n"[i == 9]);
  for (int i = 0; i < B * O; ++i)
    he += fabs(Y[i] - float(hY[i]));
  printf("  error: %.5f\n", he / B / O);
  printf("int8:\n  ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", float(iY[i]) / 127 / 127, " \n"[i == 9]);
  for (int i = 0; i < B * O; ++i)
    ie += fabs(Y[i] - float(iY[i]) / 127 / 127);
  printf("  error: %.5f\n", ie / B / O);

  free_memory(iX, iW, iY);
  free_memory(fX, fW, fY);
  free_memory(hX, hW, hY);
  cudaFree(Y);
  cudaFree(fW_T);
  return 0;
}
