#pragma once

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <thrust/device_vector.h>

#include "src/custom/transformer/proto/gpt.pb.h"
#include "src/custom/transformer/util.h"

namespace lab {
namespace nmt {

/*
    extract the gpt network weights from proto to gpu device memory
*/
template <OperationType OpType_> class GptWeight {
private:
  typedef OperationTypeTraits<OpType_> _optraits;
  typedef typename _optraits::DataType _DataType;
  _DataType float2required(float value);
  void get_model_config(const Gpt &gpt);
  std::string parse_emb_wei(const EmbeddingLayer &layer);
  std::string parse_enc_wei(const Gpt &gpt);

  std::vector<const _DataType *> _p_d_src_emb_wei; // size: 4
  std::vector<const _DataType *> _p_d_enc_wei;     // size: 12 * enc_layer_num

  // store the weights on gpu memo
  thrust::device_vector<_DataType> _d_src_emb_wei;
  thrust::device_vector<_DataType> _d_enc_wei;

public:
  std::string initializing(std::string proto_path);

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

  int _hidden_size;
  int _inner_size;
  int _max_step;
  int _src_vocab_size;
  int _n_enc_layer;  // number of encoder layer
  int _dim_per_head;
  int _weight_per_enc_layer;  // 12

  int _head_num;
  int _padding_id;  // for src
};

}  // namespace nmt
}  // namespace lab
