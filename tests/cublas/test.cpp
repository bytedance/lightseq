#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>
#include "gemm.h"

void _main(int C, int B, int O, int H, int iteration, bool debug) {
  printf(
      ">>>>>>>>>>>>>>>>>>>> shape: X(%d, %d, %d), W(%d, %d, %d) "
      ">>>>>>>>>>>>>>>>>>>>\n",
      C, B, H, C, O, H);

  float *Y;
  if (debug) checkCudaStatus(cudaMallocManaged(&Y, C * B * O * sizeof(float)));

  float *fX, *fW, *fY;
  __half *hX, *hW, *hY;
  int8_t *iX, *iW;
  int32_t *iY;
  allocate_memory(C, B, O, H, &fX, &fW, &fY);
  allocate_memory(C, B, O, H, &hX, &hW, &hY);
  allocate_memory(C, B, O, H, &iX, &iW, &iY);

  float f_alpha = 1, f_beta = 0;
  __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
  int32_t i_alpha = 1, i_beta = 0;

  init_data(fX, hX, iX, fW, hW, iW, C, B, O, H);

  if (debug) matmul(fX, fW, Y, C, B, O, H);

  cublasHandle_t handle;
  cublasLtHandle_t lt_handle;
  checkCublasStatus(cublasCreate(&handle));
  checkCublasStatus(cublasLtCreate(&lt_handle));
  
  // test cublas
  test_gemm_ex(handle, C, B, O, H, fX, fW, fY, &f_alpha, &f_beta, 99,
               iteration);
  test_gemm_ex(handle, C, B, O, H, hX, hW, hY, &h_alpha, &h_beta, 99,
               iteration);
  test_gemm_ex(handle, C, B, O, H, iX, iW, iY, &i_alpha, &i_beta, 99,
               iteration);

  // test cublas_lt
  test_lt_matmul(lt_handle, C, B, O, H, fX, fW, fY, &f_alpha, &f_beta, iteration);
  test_lt_matmul(lt_handle, C, B, O, H, hX, hW, hY, &h_alpha, &h_beta, iteration);
  test_lt_matmul_int8(lt_handle, C, B, O, H, iX, iW, iY, &i_alpha, &i_beta,
                      iteration);

  print_res(Y, fY, hY, iY, C, B, O, H, debug);

  free_memory(fX, fW, fY);
  free_memory(hX, hW, hY);
  free_memory(iX, iW, iY);
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