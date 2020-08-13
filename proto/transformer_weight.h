#pragma once

#include <fcntl.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <thrust/device_vector.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "src/custom/byseqlib/proto/transformer.pb.h"
#include "src/custom/byseqlib/tools/util.h"

namespace byseqlib {
namespace cuda {

/*
Load the model weights which stored in custom proto file into GPU memory.
*/
template <OperationType OpType_>
class TransformerWeight {
 private:
  typedef OperationTypeTraits<OpType_> _optraits;
  typedef typename _optraits::DataType _DataType;
  _DataType float2required(float value);
  void get_model_config(const Transformer &transformer,
                        bool only_decoder = false);
  std::string parse_emb_wei(const EmbeddingLayer &layer, std::string source);
  std::string parse_enc_wei(const Transformer &transformer);
  std::string parse_dec_wei(const Transformer &transformer);

  // store the weights pointer
  std::vector<const _DataType *> _p_d_src_emb_wei;  // size: 4
  std::vector<const _DataType *> _p_d_trg_emb_wei;  // size: 4
  std::vector<const _DataType *> _p_d_enc_wei;      // size: 12 * enc_layer_num
  std::vector<const _DataType *> _p_d_dec_wei;      // size: 18 * dec_layer_num

  // store the weights on gpu memo
  thrust::device_vector<_DataType> _d_src_emb_wei;
  thrust::device_vector<_DataType> _d_trg_emb_wei;
  thrust::device_vector<_DataType> _d_enc_wei;
  thrust::device_vector<_DataType> _d_dec_wei;

 public:
  std::string initializing(std::string proto_path, bool only_decoder = false);

  const std::vector<const _DataType *> &get_src_emb_wei() const {
    // {token_emb, pos_emb, norm_scale, norm_bias}
    return _p_d_src_emb_wei;
  }

  const std::vector<const _DataType *> &get_trg_emb_wei() const {
    // {token_emb, pos_emb, norm_scale, norm_bias, encdec_kv_kernel,
    // encdec_kv_bias, logit_bias}
    return _p_d_trg_emb_wei;
  }

  const std::vector<const _DataType *> &get_enc_wei() const {
    // {multihead_norm_scale, multihead_norm_bias, multihead_qkv_kernel,
    // multihead_qkv_bias multihead_output_kernel, multihead_output_bias
    // ffn_norm_scale, ffn_norm_bias}
    // ffn_first_kernel, ffn_first_bias, ffn_second_kernel, ffn_second_bias} *
    // encoder_layer_num
    return _p_d_enc_wei;
  }

  const std::vector<const _DataType *> &get_dec_wei() const {
    // {self_norm_scale, self_norm_bias,
    // self_qkv_kernel, self_qkv_bias, self_output_kernel, self_output_bias,
    // encdec_norm_scale, encdec_norm_bias,
    // encdec_q_kernel, encdec_q_bias, encdec_output_kernel,  encdec_output_bias
    // ffn_norm_scale, ffn_norm_bias, ffn_first_kernel, ffn_first_bias,
    // ffn_second_kernel, ffn_second_bias, } * decoder_layer_num
    return _p_d_dec_wei;
  }

  int _hidden_size;
  int _inner_size;
  int _max_step;
  int _src_vocab_size;
  int _trg_vocab_size;
  int _n_enc_layer;  // number of encoder layer
  int _n_dec_layer;  // number of decoder layer
  int _dim_per_head;
  int _weight_per_enc_layer;  // 12
  int _weight_per_dec_layer;  // 18

  int _head_num;
  int _beam_size;
  int _extra_decode_length;
  float _length_penalty;
  int _padding_id;  // for src
  int _start_id;    // for trg
  float _diverse_lambda;
  std::string _sampling_method;
  int _topk;
  float _topp;
};

}  // namespace cuda
}  // namespace byseqlib
