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

class Decoder {
 public:
  Decoder(int max_batch_size, const int* p_d_padding_mask,
          const float* p_d_encoder_output, int* p_d_result,
          const TransformerWeight& tw, cublasHandle_t hd);
  int compute_buffer_bytesize();
  void init_buffer(void* pbuf);
  std::string check();
  void run_one_infer(int batch_size, int batch_seq_len);
  int _cur_step;

 private:
  // private mem function
  void project_encoder_output();
  bool run_step();
  void embedding();
  void decoder_stack();
  void self_attention();
  void encdec_attention();
  void ffn_add_norm();
  bool beam_search();
  void update_new_seq_probs();

  // constructor init var
  const int _max_batch_size;
  const int _max_thread_per_block;
  int _h_can_num_batch;
  size_t _cub_sort_buffer_bytes;
  const TransformerWeight& _tw;
  cublasHandle_t _hd;

  const int* _p_d_padding_mask;
  const float* _p_d_encoder_output;
  int* _p_d_result;

  std::vector<float> _h_alive_seq_probs;
  std::vector<float> _h_finished_scores;
  std::vector<float> _h_length_norm;
  float* _p_d_alive_seq_probs;
  float* _p_d_can_probs;
  int* _p_d_can_idx;
  int* _p_d_finished_beam_tag;
  float* _p_d_finished_scores;
  int* _p_d_can_num;
  int* _p_d_alive_seq;
  int* _p_d_alive_seq_buf;
  int* _p_d_finished_seq;
  float* _p_d_cur_step_query;
  // cur step's projected query-key-value in self atten, one pointer for one
  // decoder layer device memory in [batch_size, beam_size, 3, hidden_size]
  // format
  float* _p_d_self_step_qkv;
  // key re-arrange for batch_geem in self atten, one pointer for one decoder
  // layer device memory in [batch_size, beam_size, head_num, dim_per_head,
  // max_step] format
  std::vector<float*> _p_d_self_k_bgeem;
  float** _p_d_self_k_bgeem1;
  float** _p_d_self_k_bgeem2;
  // value re-arrange for batch_geem in self atten, one pointer for one decoder
  // layer device memory in [batch_size, beam_size, head_num, max_step,
  // dim_per_head] format
  std::vector<float*> _p_d_self_v_bgeem;
  float** _p_d_self_v_bgeem1;
  float** _p_d_self_v_bgeem2;
  // key re-arrange for batch_geem in encdec atten, one pointer for one decoder
  // layer device memory in [batch_size, head_num, dim_per_head, batch_seq_len]
  // format
  std::vector<float*> _p_d_encdec_k_bgeem;
  // value re-arrange for batch_geem in encdec atten, one pointer for one
  // decoder layer device memory in [batch_size, head_num, batch_seq_len,
  // dim_per_head] format
  std::vector<float*> _p_d_encdec_v_bgeem;
  float* _p_d_query_buf1;
  float* _p_d_query_buf2;
  float* _p_d_c;
  float* _p_d_encoder_out_buf;
  float* _p_d_logit_buf;

  int _batch_size;
  int _batch_seq_len;
  int _batch_token_num;
  int _layer_id;
  int _weight_offset;
  int _step_token_num;
  int _batch_max_decode_length;

  const std::vector<const float*>& _p_d_trg_emb_wei;  // size: 7
  const std::vector<const float*>& _p_d_dec_wei;  // size: 18 * dec_layer_num
  const float _fone;
  const float _fzero;
  const float _atten_scaler;   // scaling factor of Scaled Dot-Product Attention
  const float _output_scaler;  // output scaling factor of the liner project
                               // after decoder
  const int _layer_size_encdec_k;
  const int _layer_size_self_k;
};

}  // namespace nmt
}  // namespace lab
