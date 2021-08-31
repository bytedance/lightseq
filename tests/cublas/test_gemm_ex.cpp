#include <sys/time.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
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

void matmul(float *A, float *B, float *C, int m, int n, int k) {
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      C[i * n + j] = 0;
      for (int kk = 0; kk < k; ++kk)
        C[i * n + j] += A[i * k + kk] * B[j * k + kk];
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
                   S *C, int lda, int ldb, int ldc, cudaDataType_t AType,
                   cudaDataType_t BType, cudaDataType_t CType,
                   cudaDataType_t ComputeType, S *alpha, S *beta,
                   cublasGemmAlgo_t algo) {
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
    return;
  }

  float total_time = 0;
  for (int i = 0; i < iteration; ++i) {
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    int success =
        cublas_gemm_ex(handle, CUBLAS_OP_T, CUBLAS_OP_N, O, B, H, W, X, Y, H, H,
                       O, AType, BType, CType, ComputeType, alpha, beta,
                       static_cast<cublasGemmAlgo_t>(algo));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (success > 0 && i >= 10) total_time += time;
  }
  if (total_time > 0)
    printf("algo %d: %.3f ms\n", algo, total_time / (iteration - 10));
}

void _main(int B, int O, int H, int iteration, bool debug) {
  printf(
      ">>>>>>>>>>>>>>>>>>>> shape: X(%d, %d), W(%d, %d) >>>>>>>>>>>>>>>>>>>>\n",
      B, H, O, H);
  int start_algo = CUBLAS_GEMM_DEFAULT;
  int end_algo = CUBLAS_GEMM_ALGO23;
  int start_algo_t_op = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  int end_algo_t_op = CUBLAS_GEMM_ALGO15_TENSOR_OP;

  float *Y;
  if (debug) cudaMallocManaged(&Y, B * O * sizeof(float));

  float *fX, *fW, *fY;
  __half *hX, *hW, *hY;
  int8_t *iX, *iW;
  int32_t *iY;
  allocate_memory(B, O, H, &fX, &fW, &fY);
  allocate_memory(B, O, H, &hX, &hW, &hY);
  allocate_memory(B, O, H, &iX, &iW, &iY);

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
  if (debug) matmul(fX, fW, Y, B, O, H);

  cublasHandle_t handle;
  cublasCreate(&handle);

  printf(">>>>> test fp32 >>>>>\n");
  test_gemm_ex(handle, B, O, H, fX, fW, fY, &f_alpha, &f_beta, -1, iteration);
  test_gemm_ex(handle, B, O, H, fX, fW, fY, &f_alpha, &f_beta, 99, iteration);

  printf(">>>>> test fp16 >>>>>\n");
  test_gemm_ex(handle, B, O, H, hX, hW, hY, &h_alpha, &h_beta, -1, iteration);
  test_gemm_ex(handle, B, O, H, hX, hW, hY, &h_alpha, &h_beta, 99, iteration);

  printf(">>>>> test int8 >>>>>\n");
  test_gemm_ex(handle, B, O, H, iX, iW, iY, &i_alpha, &i_beta, -1, iteration);
  test_gemm_ex(handle, B, O, H, iX, iW, iY, &i_alpha, &i_beta, 99, iteration);

  float fe = 0, he = 0, ie = 0;
  printf(">>>>> compare result >>>>>\n");
  if (debug) {
    printf("oracle:\n  ");
    for (int i = 0; i < 10; ++i) printf("%.5f%c", Y[i], " \n"[i == 9]);
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

  free_memory(iX, iW, iY);
  free_memory(fX, fW, fY);
  free_memory(hX, hW, hY);
  if (debug) cudaFree(Y);
}

int main() {
  int iteration = 50;
  bool debug = false;
  std::vector<int> Bs = {8, 16, 4096};
  std::vector<int> Os = {1024, 3072, 4096};
  std::vector<int> Hs = {1024, 4096};
  for (int i = 0; i < Bs.size(); ++i)
    for (int j = 0; j < Os.size(); ++j)
      for (int k = 0; k < Hs.size(); ++k)
        _main(Bs[i], Os[j], Hs[k], iteration, debug);
  return 0;
}
