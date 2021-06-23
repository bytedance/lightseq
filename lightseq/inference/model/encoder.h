#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

#include "../proto/transformer_weight.h"
#include "../tools/util.h"

/**
@file
Transformer decoder, composed by gemm lib and
  custom cuda kernel function
*/

namespace lightseq {
namespace cuda {

template <OperationType OpType_>
class Encoder {
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
  const int *_p_d_token_id;  // input token id [batch_size, batch_seq_len]
  int *_p_d_padding_mask;  // true sequence length(remove padding), [batch_size]
  _DataType
      *_p_d_output;  // encoder output, [batch_size, batch_seq_len, hidden_size]
  const TransformerWeight<OpType_> &_tw;
  cudaStream_t _stream;
  cublasHandle_t _hd;
  const _DataType _fone;
  const _DataType _fzero;
  const _DataType _atten_scaler;
  const int _max_batch_dim;
  const int _max_thread_per_block;

  _DataType *_p_d_qkv_projected;
  _DataType *_p_d_q;
  _DataType *_p_d_k;
  _DataType *_p_d_v;
  _DataType *_p_d_c;
  _DataType *_p_d_ffn_buf1;
  _DataType *_p_d_ffn_buf2;

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
  Encoder(int max_batch_size, const int *p_d_token_id, int *p_d_padding_mask,
          _DataType *p_d_output, const TransformerWeight<OpType_> &tw,
          cudaStream_t stream, cublasHandle_t hd);
  long compute_buffer_bytesize();
  void init_buffer(void *pbuf);
  std::string check();
  void run_one_infer(int batch_size, int batch_seq_len);
};

}  // namespace cuda
}  // namespace lightseq
