#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>

#include "src/custom/byseqlib/proto/gpt_weight.h"
#include "src/custom/byseqlib/tools/util.h"

namespace byseqlib {
namespace cuda {

template <OperationType OpType_>
class GptEncoder {
 private:
  typedef OperationTypeTraits<OpType_> _optraits;
  typedef typename _optraits::DataType _DataType;
  const cudaDataType_t _computeType = _optraits::computeType;
  const cudaDataType_t _AType = _optraits::AType;
  const cudaDataType_t _BType = _optraits::BType;
  const cudaDataType_t _CType = _optraits::CType;

  // private member function
  void self_attention();
  void ffn_add_norm();

  const int _max_batch_size;
  const int *_p_d_token_id;  // input token id, [batch_size, batch_seq_len]
  float *_p_d_ppl;           // ppl for every seq, [batch_size]
  int *_p_d_sample_id;
  const GptWeight<OpType_> &_tw;
  cudaStream_t _stream;
  cublasHandle_t _hd;
  const _DataType _fone;
  const _DataType _fzero;
  const _DataType _atten_scaler;
  const int _max_batch_dim;
  const int _max_thread_per_block;
  std::vector<int> _h_real_seq_len;
  std::vector<float> _h_ppl;
  std::vector<int> _h_sample_id;
  int* _p_d_unfinished;
  int _h_unfinished;

  // gpu memeory buffer
  _DataType *_p_d_query;
  _DataType *_p_d_qkv_projected;
  _DataType *_p_d_q;
  _DataType *_p_d_k;
  _DataType *_p_d_v;
  _DataType *_p_d_c;
  _DataType *_p_d_ffn_buf1;
  _DataType *_p_d_ffn_buf2;
  _DataType *_p_d_logit;
  int *_p_d_real_seq_len;         // [batch_size]
  int *_p_d_sample_id_buf;        // [batch_size, max_step]
  curandState *_p_d_curandstate;  //[batch_size]

  // {token_emb, pos_emb, norm_scale, norm_bias}
  const std::vector<const _DataType *> &_p_d_src_emb_wei;
  // {multihead_norm_scale, multihead_norm_bias, multihead_qkv_kernel,
  // multihead_qkv_bias multihead_output_kernel, multihead_output_bias
  // ffn_norm_scale, ffn_norm_bias}
  // ffn_first_kernel, ffn_first_bias, ffn_second_kernel, ffn_second_bias} *
  // encoder_layer_num
  const std::vector<const _DataType *> &_p_d_enc_wei;

  int _batch_size;
  int _batch_seq_len;
  int _batch_token_num;
  int _layer_id;
  int _weight_offset;

 public:
  GptEncoder(int max_batch_size, const int *p_d_token_id, float *p_d_ppl,
             int *p_d_sample_id, const GptWeight<OpType_> &tw,
             cudaStream_t stream, cublasHandle_t hd);
  int compute_buffer_bytesize();
  void init_buffer(void *pbuf);
  std::string check();
  void run_one_infer(int batch_size, int batch_seq_len);
  void run_one_sample(int batch_size, int batch_seq_len);
  int sample_one_token();
  void compute_ppl();
};

}  // namespace cuda
}  // namespace byseqlib
