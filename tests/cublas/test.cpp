#include "gemm.h"

void _main(std::string name, int C, int B, int O, int H, int iteration,
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
  float lt_ft = -1, lt_ht = -1;
  float lt_col_it = -1, lt_col4_4r2_8c_it = -1, lt_col32_2r_4r4_it = -1;

  printf(">>>>> test cublas gemm ex >>>>>\n");
  cublas_ft = test_gemm_ex(handle, C, B, O, H, fX, fW, fY, &f_alpha, &f_beta,
                           iteration);
  print_res(fY, fY, cublas_ft, C, B, O, H, "cublas fp32", false, debug);
  cublas_ht = test_gemm_ex(handle, C, B, O, H, hX, hW, hY, &h_alpha, &h_beta,
                           iteration);
  print_res(fY, hY, cublas_ht, C, B, O, H, "cublas fp16", false, debug);
  cublas_it = test_gemm_ex(handle, C, B, O, H, iX, iW, iY, &i_alpha, &i_beta,
                           iteration);
  print_res(fY, iY, cublas_it, C, B, O, H, "cublas int8", true, debug);

  if (C == 1) {
    printf(">>>>> test cublas lt matmul >>>>>\n");
    lt_ft = test_lt_matmul(lt_handle, C, B, O, H, fX, fW, fY, &f_alpha, &f_beta,
                           iteration);
    print_res(fY, fY, lt_ft, C, B, O, H, "lt fp32", false, debug);
    lt_ht = test_lt_matmul(lt_handle, C, B, O, H, hX, hW, hY, &h_alpha, &h_beta,
                           iteration);
    print_res(fY, hY, lt_ht, C, B, O, H, "lt fp16", false, debug);
    lt_col_it = test_lt_matmul_int8(lt_handle, C, B, O, H, iX, iW, iY, &i_alpha,
                                    &i_beta, iteration, 0);
    print_res(fY, iY, lt_col_it, C, B, O, H, "lt_col int8", true, debug);
    lt_col4_4r2_8c_it = test_lt_matmul_int8(lt_handle, C, B, O, H, iX, iW, iY,
                                            &i_alpha, &i_beta, iteration, 1);
    print_res(fY, iY, lt_col4_4r2_8c_it, C, B, O, H, "lt_col4_4r2_8c int8",
              true, debug);
    lt_col32_2r_4r4_it = test_lt_matmul_int8(lt_handle, C, B, O, H, iX, iW, iY,
                                             &i_alpha, &i_beta, iteration, 2);
    print_res(fY, iY, lt_col32_2r_4r4_it, C, B, O, H, "lt_col32_2r_4r4 int8",
              true, debug);
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

  free_memory(fX, fW, fY);
  free_memory(hX, hW, hY);
  free_memory(iX, iW, iY);
  if (debug) checkCudaStatus(cudaFree(Y));
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

  int inner_size = 4 * hidden_size;
  int head_dim = hidden_size / head_num;
  int batch_token_size = batch_size * seq_len;
  int batch_beam_size = batch_size * beam_size;
  int batch_head_num = batch_size * head_num;

  vpsvi shapes;
  // encoder
  vec_pb(shapes, "enc attn qkv",
         {1, batch_token_size, 3 * hidden_size, hidden_size});
  vec_pb(shapes, "enc attn out",
         {1, batch_token_size, hidden_size, hidden_size});
  vec_pb(shapes, "enc ffn1", {1, batch_token_size, inner_size, hidden_size});
  vec_pb(shapes, "enc ffn2", {1, batch_token_size, hidden_size, inner_size});
  // decoder
  vec_pb(shapes, "dec attn qkv",
         {1, batch_beam_size, 3 * hidden_size, hidden_size});
  vec_pb(shapes, "dec attn out",
         {1, batch_beam_size, hidden_size, hidden_size});
  vec_pb(shapes, "dec ffn1", {1, batch_beam_size, inner_size, hidden_size});
  vec_pb(shapes, "dec ffn2", {1, batch_beam_size, hidden_size, inner_size});
  // logits
  vec_pb(shapes, "logits", {1, batch_beam_size, vocab_size, hidden_size});
  // batch gemm (encoder)
  vec_pb(shapes, "enc attn score",
         {batch_head_num, seq_len, seq_len, head_dim});
  vec_pb(shapes, "enc attn value",
         {batch_head_num, head_dim, seq_len, seq_len});
  // batch gemm (decoder encdec attention)
  vec_pb(shapes, "dec encdec attn score",
         {batch_head_num, seq_len, beam_size, head_dim});
  vec_pb(shapes, "dec encdec attn value",
         {batch_head_num, head_dim, beam_size, seq_len});
  // batch gemm (decoder self attention)
  for (int step = 1; step <= seq_len; step += 10) {
    vec_pb(shapes, "dec self attn score",
           {batch_beam_size * head_num, step, 1, head_dim});
    vec_pb(shapes, "dec self attn value",
           {batch_beam_size * head_num, head_dim, 1, step});
  }

  for (auto shape : shapes) {
    _main(shape.first, shape.second[0], shape.second[1], shape.second[2],
          shape.second[3], iteration, debug);
  }

  return 0;
}
