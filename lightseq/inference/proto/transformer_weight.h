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

#include "../tools/util.h"
#include "transformer.pb.h"

namespace lightseq {
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

  // parsing function for protobuffer
  void proto_get_model_config(const Transformer &transformer,
                              bool only_decoder = false);
  std::string proto_parse_emb_wei(const EmbeddingLayer &layer,
                                  std::string source);
  std::string proto_parse_enc_wei(const Transformer &transformer);
  std::string proto_parse_dec_wei(const Transformer &transformer);

  // parsing function for hdf5
  void hdf5_get_model_config(hid_t hdf5_file, bool only_decoder = false);
  void hdf5_parse_emb_wei(hid_t hdf5_file, std::string source);
  void hdf5_parse_enc_wei(hid_t hdf5_file);
  void hdf5_parse_dec_wei(hid_t hdf5_file);

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
  thrust::device_vector<int> _d_trg_vocab_mask;
  thrust::device_vector<_DataType> _d_src_lang_emb;
  thrust::device_vector<_DataType> _d_trg_lang_emb;

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
  int _end_id;
  float _diverse_lambda;
  std::string _sampling_method;
  int _topk;
  float _topp;
  bool _is_post_ln;
  bool _no_scale_embedding;
  bool _use_gelu;
  bool _is_multilingual;
  const int *_p_d_trg_vocab_mask;

  void print_model_config() {
    std::cout << "***model config***" << std::endl;
    std::cout << "encoder layers: " << _n_enc_layer << std::endl;
    std::cout << "decoder layers: " << _n_dec_layer << std::endl;
    std::cout << "hidden size: " << _hidden_size << std::endl;
    std::cout << "inner size: " << _inner_size << std::endl;
    std::cout << "head number: " << _head_num << std::endl;
    std::cout << "dim per head: " << _dim_per_head << std::endl;
    std::cout << "src vocab size: " << _src_vocab_size << std::endl;
    std::cout << "trg vocab size: " << _trg_vocab_size << std::endl;
    std::cout << "is_post_ln: " << _is_post_ln << std::endl;
    std::cout << "no_scale_embedding: " << _no_scale_embedding << std::endl;
    std::cout << "use_gelu: " << _use_gelu << std::endl;
    std::cout << "start_id: " << _start_id << std::endl;
    std::cout << "end_id: " << _end_id << std::endl;
    std::cout << "padding_id: " << _padding_id << std::endl;
    std::cout << "is_multilingual: " << _is_multilingual << std::endl;
    std::cout << std::endl;
    std::cout << "***generator config***" << std::endl;
    std::cout << "beam size: " << _beam_size << std::endl;
    std::cout << "extra decode length(max decode length - src input length): "
              << _extra_decode_length << std::endl;
    std::cout << "length penalty: " << _length_penalty << std::endl;
    std::cout << "diverse lambda: " << _diverse_lambda << std::endl;
    std::cout << "sampling method: " << _sampling_method << std::endl;
    std::cout << "topk: " << _topk << std::endl;
    std::cout << "topp: " << _topp << std::endl;
  }
};

}  // namespace cuda
}  // namespace lightseq
