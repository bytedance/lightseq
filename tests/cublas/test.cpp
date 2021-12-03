#include <vector>
#include "gemm.h"

typedef std::vector<int> vi;
typedef std::vector<vi> vvi;
typedef std::vector<float> vf;

vf _main(int C, int B, int O, int H, int iteration, bool debug) {
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

  printf(">>>>> test cublas gemm ex >>>>>\n");
  float cublas_ft = test_gemm_ex(handle, C, B, O, H, fX, fW, fY, &f_alpha,
                                 &f_beta, iteration);
  float cublas_ht = test_gemm_ex(handle, C, B, O, H, hX, hW, hY, &h_alpha,
                                 &h_beta, iteration);
  float cublas_it = test_gemm_ex(handle, C, B, O, H, iX, iW, iY, &i_alpha,
                                 &i_beta, iteration);
  print_res(Y, fY, hY, iY, C, B, O, H, cublas_ft, cublas_ht, cublas_it, debug);

  printf(">>>>> test cublas lt matmul >>>>>\n");
  float cublaslt_ft = test_lt_matmul(lt_handle, C, B, O, H, fX, fW, fY,
                                     &f_alpha, &f_beta, iteration);
  float cublaslt_ht = test_lt_matmul(lt_handle, C, B, O, H, hX, hW, hY,
                                     &h_alpha, &h_beta, iteration);
  float cublaslt_it = test_lt_matmul_int8(lt_handle, C, B, O, H, iX, iW, iY,
                                          &i_alpha, &i_beta, iteration);
  print_res(Y, fY, hY, iY, C, B, O, H, cublaslt_ft, cublaslt_ht, cublaslt_it,
            debug);

  // printf(">>>>> test tvm gemm >>>>>\n");
  // float tvm_it = test_tvm_gemm(iX, iW, iY, iteration);
  // if (debug)
  //   for (int i = 0; i < 10; ++i)
  //     printf("%.5f%c", float(iY[i]) / 127 / 127, " \n"[i == 9]);
  // float ie = 0;
  // for (int i = 0; i < C * B * O; ++i)
  //   ie += fabs((debug ? Y[i] : fY[i]) - float(iY[i]) / 127 / 127);
  // printf("  diff: %.5f\n", ie / C / B / O);
  // printf("  time: %.3f ms\n", tvm_it);

  printf("SPEEDUP (cublas fp16 / lt fp16):     %.3f\n",
         cublas_ht / cublaslt_ht);
  printf("SPEEDUP (cublas fp16 / cublas int8): %.3f\n", cublas_ht / cublas_it);
  printf("SPEEDUP (cublas fp16 / lt int8):     %.3f\n",
         cublas_ht / cublaslt_it);

  free_memory(fX, fW, fY);
  free_memory(hX, hW, hY);
  free_memory(iX, iW, iY);
  if (debug) checkCudaStatus(cudaFree(Y));

  return {cublas_ht / cublaslt_ht, cublas_ht / cublas_it,
          cublas_ht / cublaslt_it};
}

int main() {
  int iteration = 10;
  bool debug = false;

  int batch_size = 8;
  int seq_len = 70;
  int beam_size = 4;
  int hidden_size = 1024;
  int vocab_size = 46896;

  vvi shapes;
  // encoder
  shapes.push_back({1, batch_size * seq_len, 3 * hidden_size, hidden_size});
  shapes.push_back({1, batch_size * seq_len, hidden_size, hidden_size});
  shapes.push_back({1, batch_size * seq_len, 4 * hidden_size, hidden_size});
  shapes.push_back({1, batch_size * seq_len, hidden_size, 4 * hidden_size});
  // decoder
  shapes.push_back({1, batch_size * beam_size, 3 * hidden_size, hidden_size});
  shapes.push_back({1, batch_size * beam_size, hidden_size, hidden_size});
  shapes.push_back({1, batch_size * beam_size, 4 * hidden_size, hidden_size});
  shapes.push_back({1, batch_size * beam_size, hidden_size, 4 * hidden_size});
  // logits
  shapes.push_back({1, batch_size * beam_size, vocab_size, hidden_size});

  vf speedup = vf(3, 0);
  for (auto shape : shapes) {
    vf su = _main(shape[0], shape[1], shape[2], shape[3], iteration, debug);
    for (int i = 0; i < 3; ++i) speedup[i] += su[i];
  }

  printf(">>>>>>>>>>>>>>>>>>>> SUMMARY >>>>>>>>>>>>>>>>>>>>\n");
  printf("AVERAGE SPEEDUP (cublas fp16 / lt fp16):     %.3f\n",
         speedup[0] / shapes.size());
  printf("AVERAGE SPEEDUP (cublas fp16 / cublas int8): %.3f\n",
         speedup[1] / shapes.size());
  printf("AVERAGE SPEEDUP (cublas fp16 / lt int8):     %.3f\n",
         speedup[2] / shapes.size());
  return 0;
}
