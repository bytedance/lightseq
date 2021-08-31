#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>
#include "util.h"

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
int cublas_gemm_strided_batched_ex(
    cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB,
    int bsz, int m, int n, int k, T *A, T *B, S *C, int lda, int ldb, int ldc,
    cudaDataType_t AType, cudaDataType_t BType, cudaDataType_t CType,
    cudaDataType_t ComputeType, S *alpha, S *beta, cublasGemmAlgo_t algo) {
  cublasStatus_t status;
  int64_t strideA = m * k, strideB = k * n, strideC = m * n;
  status = cublasGemmStridedBatchedEx(
      handle, transA, transB, m, n, k, alpha, A, AType, lda, strideA, B, BType,
      ldb, strideB, beta, C, CType, ldc, strideC, bsz, ComputeType, algo);

  if (status == CUBLAS_STATUS_SUCCESS)
    return 1;
  else
    return -1;
}

template <typename T, typename S>
void test_gemm_ex(cublasHandle_t handle, int C, int B, int O, int H, T *X, T *W,
                  S *Y, S *alpha, S *beta, int algo, int iteration) {
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
    int success;
    if (C > 1)
      cublas_gemm_strided_batched_ex(handle, CUBLAS_OP_T, CUBLAS_OP_N, C, O, B,
                                     H, W, X, Y, H, H, O, AType, BType, CType,
                                     ComputeType, alpha, beta,
                                     static_cast<cublasGemmAlgo_t>(algo));
    else
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

void _main(int C, int B, int O, int H, int iteration, bool debug) {
  printf(
      ">>>>>>>>>>>>>>>>>>>> shape: X(%d, %d, %d), W(%d, %d, %d) "
      ">>>>>>>>>>>>>>>>>>>>\n",
      C, B, H, C, O, H);

  float *Y;
  float *fX, *fW, *fY;
  __half *hX, *hW, *hY;
  int8_t *iX, *iW;
  int32_t *iY;
  if (debug) checkCudaStatus(cudaMallocManaged(&Y, C * B * O * sizeof(float)));
  allocate_memory(C, B, O, H, &fX, &fW, &fY);
  allocate_memory(C, B, O, H, &hX, &hW, &hY);
  allocate_memory(C, B, O, H, &iX, &iW, &iY);

  float f_alpha = 1, f_beta = 0;
  __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
  int32_t i_alpha = 1, i_beta = 0;

  init_data(fX, hX, iX, fW, hW, iW, C, B, O, H);

  if (debug) matmul(fX, fW, Y, C, B, O, H);

  cublasHandle_t handle;
  checkCublasStatus(cublasCreate(&handle));

  printf(">>>>> test fp32 >>>>>\n");
  test_gemm_ex(handle, C, B, O, H, fX, fW, fY, &f_alpha, &f_beta, -1,
               iteration);
  test_gemm_ex(handle, C, B, O, H, fX, fW, fY, &f_alpha, &f_beta, 99,
               iteration);

  printf(">>>>> test fp16 >>>>>\n");
  test_gemm_ex(handle, C, B, O, H, hX, hW, hY, &h_alpha, &h_beta, -1,
               iteration);
  test_gemm_ex(handle, C, B, O, H, hX, hW, hY, &h_alpha, &h_beta, 99,
               iteration);

  printf(">>>>> test int8 >>>>>\n");
  test_gemm_ex(handle, C, B, O, H, iX, iW, iY, &i_alpha, &i_beta, -1,
               iteration);
  test_gemm_ex(handle, C, B, O, H, iX, iW, iY, &i_alpha, &i_beta, 99,
               iteration);

  print_res(Y, fY, hY, iY, C, B, O, H, debug);

  free_memory(iX, iW, iY);
  free_memory(fX, fW, fY);
  free_memory(hX, hW, hY);
  if (debug) checkCudaStatus(cudaFree(Y));
}

int main() {
  int iteration = 50;
  bool debug = false;
  std::vector<int> Cs = {1, 8, 64};
  std::vector<int> Bs = {8, 16, 4096};
  std::vector<int> Os = {1024, 3072, 4096};
  std::vector<int> Hs = {1024, 4096};
  for (int l = 0; l < Cs.size(); ++l)
    for (int i = 0; i < Bs.size(); ++i)
      for (int j = 0; j < Os.size(); ++j)
        for (int k = 0; k < Hs.size(); ++k)
          _main(Cs[l], Bs[i], Os[j], Hs[k], iteration, debug);
  return 0;
}
