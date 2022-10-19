#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <cublasLt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

#include "../proto/quant_gpt_weight.h"
#include "../tools/util.h"

namespace lightseq {
namespace cuda {

template <OperationType OpType_>
class QuantGptEncoder {
 private:
  typedef OperationTypeTraits<OpType_> _optraits;
  typedef typename _optraits::DataType _DataType;
  const cudaDataType_t _computeType = _optraits::computeType;
  const cudaDataType_t _AType = _optraits::AType;
  const cudaDataType_t _BType = _optraits::BType;
  const cudaDataType_t _CType = _optraits::CType;

  // private member function
  void self_attention();
  void self_attention_with_cache();
  void ffn_add_norm();
  void ffn_add_norm_with_cache();
  int sample_one_token();
  int sample_one_token_with_cache();

  const int _max_batch_size;

  const QuantGptWeight<OpType_> &_tw;
  cudaStream_t _stream;
  cudaStream_t _cache_stream;
  cublasHandle_t _hd;
  cublasLtHandle_t _cublas_lt_handle;
  const bool _sm_gt_eq_80;

  const _DataType _fone;
  const _DataType _fzero;
  const int32_t _ione;
  const int32_t _izero;
  const _DataType _atten_scaler;
  const int _max_batch_dim;
  const int _max_thread_per_block;
  std::vector<int> _h_real_seq_len;
  std::vector<float> _h_ppl;
  std::vector<int> _h_sample_id;
  int _h_unfinished;

  // gpu memory buffer
  _DataType *_p_d_query;
  _DataType *_p_d_k_cache;
  _DataType *_p_d_v_cache;
  _DataType *_p_d_qkv_projected;
  _DataType *_p_d_q;
  _DataType *_p_d_k;
  _DataType *_p_d_v;
  _DataType *_p_d_c;
  _DataType *_p_d_ffn_buf1;
  _DataType *_p_d_ffn_buf2;
  _DataType *_p_d_logit;
  int *_p_d_real_seq_len;   // [batch_size]
  int *_p_d_sample_id_buf;  // [batch_size, max_step]
  int *_p_d_last_sample_id;
  int *_p_d_unfinished;
  curandState *_p_d_curandstate;  //[batch_size]

  int8_t *_int8_ffn_in_buf;
  int32_t *_int32_ffn_out_buf;
  int8_t *_int8_ffn_out_buf;
  std::vector<int8_t *> _p_d_self_k_cache;
  std::vector<int8_t *> _p_d_self_v_cache;
  int8_t **_p_d_self_k_cache1;
  int8_t **_p_d_self_k_cache2;
  int8_t **_p_d_self_v_cache1;
  int8_t **_p_d_self_v_cache2;

  // {token_emb, pos_emb, norm_scale, norm_bias}
  const std::vector<const _DataType *> &_p_d_src_emb_wei;
  // {multihead_norm_scale, multihead_norm_bias, multihead_qkv_kernel,
  // multihead_qkv_bias multihead_output_kernel, multihead_output_bias
  // ffn_norm_scale, ffn_norm_bias}
  // ffn_first_kernel, ffn_first_bias, ffn_second_kernel, ffn_second_bias} *
  // encoder_layer_num
  const std::vector<const _DataType *> &_p_d_enc_wei;
  std::vector<const _DataType *> _p_device_wei;
  std::vector<const _DataType *> _p_device_emb;

  std::vector<int8_t *> _int8_p_d_enc_wei;
  int8_t *_int8_p_d_src_emb_wei;
  int8_t *_int8_p_d_src_emb_bottom_wei;
  const float _quant_range = 127;
  const float _src_emb_clip_max;
  const float _output_ln_clip_max;
  const float _logits_clip_max;
  const std::vector<float> _enc_clip_max;  // size: 12 * enc_layer_num
  std::vector<_DataType *> _scaled_ffn2_colsum;

  int _batch_size;
  int _batch_token_num;
  int _layer_id;
  int _weight_offset;
  bool _is_benchmark;

  const std::set<std::string> kSamplingMethods = {"topk", "topp", "ppl"};

 public:
  int _batch_seq_len;
  const int *_p_d_token_id;  // input token id, [batch_size, batch_seq_len]
  float *_p_d_ppl;           // ppl for every seq, [batch_size]
  int *_p_d_sample_id;

  QuantGptEncoder(int max_batch_size, const int *p_d_token_id, float *p_d_ppl,
                  int *p_d_sample_id, const QuantGptWeight<OpType_> &tw,
                  cudaStream_t stream, cudaStream_t cache_stream,
                  cublasHandle_t hd);
  void init_buffer();
  std::string check();
  void run_one_infer(int batch_size, int batch_seq_len);
  int run_one_sample(int batch_size, int batch_seq_len);
  void compute_ppl();
  void benchmark_mode(bool is_benchmark);
};

}  // namespace cuda
}  // namespace lightseq
