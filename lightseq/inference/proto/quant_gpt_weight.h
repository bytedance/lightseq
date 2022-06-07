#pragma once

#include <fcntl.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "quant_gpt.pb.h"
#include "../tools/util.h"

namespace lightseq {
namespace cuda {

/*
Load the model weights which stored in custom proto file into GPU memory.
*/
template <OperationType OpType_>
class QuantGptWeight {
 private:
  typedef OperationTypeTraits<OpType_> _optraits;
  typedef typename _optraits::DataType _DataType;

  _DataType float2required(float value);

  void proto_get_model_config(const QuantGpt &gpt);
  std::string proto_parse_emb_wei(const QuantGptEmbeddingLayer &layer);
  std::string proto_parse_enc_wei(const QuantGpt &gpt);

  // parsing function for hdf5
  void hdf5_get_model_config(hid_t hdf5_file);
  void hdf5_parse_emb_wei(hid_t hdf5_file);
  void hdf5_parse_enc_wei(hid_t hdf5_file);

  // store the weights pointer
  std::vector<const _DataType *> _p_d_src_emb_wei;  // size: 4
  std::vector<const _DataType *> _p_d_enc_wei;      // size: 12 * enc_layer_num

  // store the weights on gpu memory
  std::vector<_DataType> _d_src_emb_wei;
  std::vector<_DataType> _d_enc_wei;

  // store the clip_max of weights and activations
  float _src_emb_clip_max;
  float _output_ln_clip_max;
  float _logits_clip_max;
  std::vector<float> _enc_clip_max;  // size: 11 * enc_layer_num

 public:
  std::string initializing(std::string weight_path);

  const std::vector<const _DataType *> &get_src_emb_wei() const {
    // {token_emb, pos_emb, norm_scale, norm_bias}
    return _p_d_src_emb_wei;
  }

  const std::vector<const _DataType *> &get_enc_wei() const {
    // {multihead_norm_scale, multihead_norm_bias, multihead_qkv_kernel,
    // multihead_qkv_bias multihead_output_kernel, multihead_output_bias
    // ffn_norm_scale, ffn_norm_bias}
    // ffn_first_kernel, ffn_first_bias, ffn_second_kernel, ffn_second_bias} *
    // encoder_layer_num
    return _p_d_enc_wei;
  }

  float get_src_emb_clip_max() const { return _src_emb_clip_max; }

  float get_output_ln_clip_max() const { return _output_ln_clip_max; }

  float get_logits_clip_max() const { return _logits_clip_max; }

  std::vector<float> get_enc_clip_max() const { return _enc_clip_max; }

  const float _quant_range = 127;

  int _hidden_size;
  int _inner_size;
  int _max_step;
  int _src_vocab_size;
  int _n_enc_layer;  // number of encoder layer
  int _dim_per_head;
  int _weight_per_enc_layer;  // 12

  int _head_num;
  int _padding_id;  // for src
  std::string _sampling_method = "topk";
  int _topk = 4;
  float _topp = 0.75;
  int _eos_id;
};

}  // namespace cuda
}  // namespace lightseq
