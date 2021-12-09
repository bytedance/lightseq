#include <iostream>
#include <vector>
#include "gemm.h"

typedef std::vector<int> vi;
typedef std::pair<std::string, vi> psvi;
typedef std::vector<psvi> vpsvi;
typedef std::vector<float> vf;

vf _main(std::string name, int C, int B, int O, int H, int iteration,
         bool debug) {
  printf(
      ">>>>>>>>>>>>>>>>>>>> %s, shape: X(%d, %d, %d), W(%d, %d, %d) "
      ">>>>>>>>>>>>>>>>>>>>\n",
      name.c_str(), C, B, H, C, O, H);

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

  float cublas_ft = -1, cublas_ht = -1, cublas_it = -1;
  float cublaslt_ft = -1, cublaslt_ht = -1, cublaslt_it = -1;

  printf(">>>>> test cublas gemm ex >>>>>\n");
  cublas_ft = test_gemm_ex(handle, C, B, O, H, fX, fW, fY, &f_alpha, &f_beta,
                           iteration);
  cublas_ht = test_gemm_ex(handle, C, B, O, H, hX, hW, hY, &h_alpha, &h_beta,
                           iteration);
  cublas_it = test_gemm_ex(handle, C, B, O, H, iX, iW, iY, &i_alpha, &i_beta,
                           iteration);
  print_res(Y, fY, hY, iY, C, B, O, H, cublas_ft, cublas_ht, cublas_it, debug);

  if (C == 1) {
    printf(">>>>> test cublas lt matmul >>>>>\n");
    cublaslt_ft = test_lt_matmul(lt_handle, C, B, O, H, fX, fW, fY, &f_alpha,
                                 &f_beta, iteration);
    cublaslt_ht = test_lt_matmul(lt_handle, C, B, O, H, hX, hW, hY, &h_alpha,
                                 &h_beta, iteration);
    cublaslt_it = test_lt_matmul_int8(lt_handle, C, B, O, H, iX, iW, iY,
                                      &i_alpha, &i_beta, iteration);
    print_res(Y, fY, hY, iY, C, B, O, H, cublaslt_ft, cublaslt_ht, cublaslt_it,
              debug);
  }

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

  if (C == 1)
    printf("SPEEDUP (cublas fp16 / lt fp16):     %.3f\n",
           cublas_ht / cublaslt_ht);
  printf("SPEEDUP (cublas fp16 / cublas int8): %.3f\n", cublas_ht / cublas_it);
  if (C == 1)
    printf("SPEEDUP (cublas fp16 / lt int8):     %.3f\n",
           cublas_ht / cublaslt_it);

  free_memory(fX, fW, fY);
  free_memory(hX, hW, hY);
  free_memory(iX, iW, iY);
  if (debug) checkCudaStatus(cudaFree(Y));

  if (C == 1)
    return {cublas_ht / cublaslt_ht, cublas_ht / cublas_it,
            cublas_ht / cublaslt_it};
  else
    return {0, cublas_ht / cublas_it, 0};
}

int main() {
  int iteration = 10;
  bool debug = false;

  // There are some restraints of the decoder shapes
  int batch_size = 8;
  int seq_len = 70;
  int beam_size = 4;
  int hidden_size = 1024;
  int vocab_size = 46896;
  int head_num = 8;

  vpsvi shapes;
  // encoder
  shapes.push_back(std::make_pair(
      std::string("enc attn qkv"),
      vi({1, batch_size * seq_len, 3 * hidden_size, hidden_size})));
  shapes.push_back(
      std::make_pair(std::string("enc attn out"),
                     vi({1, batch_size * seq_len, hidden_size, hidden_size})));
  shapes.push_back(std::make_pair(
      std::string("enc ffn1"),
      vi({1, batch_size * seq_len, 4 * hidden_size, hidden_size})));
  shapes.push_back(std::make_pair(
      std::string("enc ffn2"),
      vi({1, batch_size * seq_len, hidden_size, 4 * hidden_size})));
  // decoder
  shapes.push_back(std::make_pair(
      std::string("dec attn qkv"),
      vi({1, batch_size * beam_size, 3 * hidden_size, hidden_size})));
  shapes.push_back(std::make_pair(
      std::string("dec attn out"),
      vi({1, batch_size * beam_size, hidden_size, hidden_size})));
  shapes.push_back(std::make_pair(
      std::string("dec ffn1"),
      vi({1, batch_size * beam_size, 4 * hidden_size, hidden_size})));
  shapes.push_back(std::make_pair(
      std::string("dec ffn2"),
      vi({1, batch_size * beam_size, hidden_size, 4 * hidden_size})));
  // logits
  shapes.push_back(
      std::make_pair(std::string("logits"),
                     vi({1, batch_size * beam_size, vocab_size, hidden_size})));
  // batch gemm (encoder)
  shapes.push_back(std::make_pair(
      std::string("enc attn score"),
      vi({batch_size * head_num, seq_len, seq_len, hidden_size / head_num})));
  shapes.push_back(std::make_pair(
      std::string("enc attn value"),
      vi({batch_size * head_num, hidden_size / head_num, seq_len, seq_len})));
  // batch gemm (decoder encdec attention)
  shapes.push_back(std::make_pair(
      std::string("dec encdec attn score"),
      vi({batch_size * head_num, seq_len, beam_size, hidden_size / head_num})));
  shapes.push_back(std::make_pair(
      std::string("dec encdec attn value"),
      vi({batch_size * head_num, hidden_size / head_num, beam_size, seq_len})));
  // batch gemm (decoder self attention)
  for (int step = 1; step <= seq_len; step += 10) {
    shapes.push_back(std::make_pair(std::string("dec self attn score"),
                                    vi({beam_size * batch_size * head_num, step,
                                        1, hidden_size / head_num})));
    shapes.push_back(std::make_pair(std::string("dec self attn value"),
                                    vi({beam_size * batch_size * head_num,
                                        hidden_size / head_num, 1, step})));
  }

  vf speedup = vf(3, 0);
  for (auto shape : shapes) {
    vf su = _main(shape.first, shape.second[0], shape.second[1],
                  shape.second[2], shape.second[3], iteration, debug);
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
