#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>

#include "src/custom/transformer/proto/transformer_weight.h"

namespace lab {
namespace nmt {

class Encoder {
 public:
  Encoder(int max_batch_size, const int* p_d_token_id, int* p_d_padding_mask,
          float* p_d_output, const TransformerWeight& tw, cublasHandle_t hd);
  int compute_buffer_bytesize();
  void init_buffer(void* pbuf);
  std::string check();
  void run_one_infer(int batch_size, int batch_seq_len);

 private:
  // private member function
  void self_attention();
  void ffn_add_norm();

  const int _max_batch_size;
  const int* _p_d_token_id;  // input token id [batch_size, batch_seq_len]
  int* _p_d_padding_mask;  // true sequence length(remove padding), [batch_size]
  float*
      _p_d_output;  // encoder output, [batch_size, batch_seq_len, hidden_size]
  const TransformerWeight& _tw;
  cublasHandle_t _hd;
  const float _fone;
  const float _fzero;
  const float _atten_scaler;
  const int _max_batch_dim;
  const int _max_thread_per_block;

  float* _p_d_qkv_projected;
  float* _p_d_q;
  float* _p_d_k;
  float* _p_d_v;
  float* _p_d_c;
  float* _p_d_ffn_buf1;
  float* _p_d_ffn_buf2;

  // {token_emb, pos_emb, norm_scale, norm_bias}
  const std::vector<const float*>& _p_d_src_emb_wei;
  // {multihead_norm_scale, multihead_norm_bias, multihead_qkv_kernel,
  // multihead_qkv_bias multihead_output_kernel, multihead_output_bias
  // ffn_norm_scale, ffn_norm_bias}
  // ffn_first_kernel, ffn_first_bias, ffn_second_kernel, ffn_second_bias} *
  // encoder_layer_num
  const std::vector<const float*>& _p_d_enc_wei;

  int _batch_size;
  int _batch_seq_len;
  int _batch_token_num;
  int _layer_id;
  int _weight_offset;
};

}  // namespace nmt
}  // namespace lab
