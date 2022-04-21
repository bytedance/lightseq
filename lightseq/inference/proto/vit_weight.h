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

#include "vit.pb.h"
#include "../tools/util.h"

namespace lightseq {
namespace cuda {

/*
Load the model weights which stored in custom proto file into GPU memory.
*/
template <OperationType OpType_>
class VitWeight {
 private:
  typedef OperationTypeTraits<OpType_> _optraits;
  typedef typename _optraits::DataType _DataType;
  _DataType float2required(float value);
  void proto_get_model_config(const Vit &vit);
  std::string proto_parse_emb_wei(const VitEmbeddingLayer &layer);
  std::string proto_parse_enc_wei(const Vit &vit);

  void hdf5_get_model_config(hid_t hdf5_file);
  void hdf5_parse_emb_wei(hid_t hdf5_file);
  void hdf5_parse_enc_wei(hid_t hdf5_file);
  // store the weights pointer
  std::vector<const _DataType *> _p_d_src_emb_wei;  // size: 6
  std::vector<const _DataType *> _p_d_enc_wei;      // size: 12 * enc_layer_num

  // store the weights on gpu memory
  thrust::device_vector<_DataType> _d_src_emb_wei;
  thrust::device_vector<_DataType> _d_enc_wei;

 public:
  std::string initializing(std::string proto_path);

  const std::vector<const _DataType *> &get_src_emb_wei() const {
    // {conv_weight, conv_bias, pos_emb, cls_embedding}
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
  int _n_enc_layer;  // number of encoder layer
  int _dim_per_head;
  int _weight_per_enc_layer;  // 12
  int _image_size;
  int _patch_size;
  int _channel_input;

  int _head_num;
  bool _is_post_ln;
  bool _use_gelu;

  void print_model_config() {
    std::cout << "***model config***" << std::endl;
    std::cout << "encoder layers: " << _n_enc_layer << std::endl;
    std::cout << "hidden size: " << _hidden_size << std::endl;
    std::cout << "inner size: " << _inner_size << std::endl;
    std::cout << "head number: " << _head_num << std::endl;
    std::cout << "dim per head: " << _dim_per_head << std::endl;
    std::cout << "use_gelu: " << _use_gelu << std::endl;
    std::cout << "is_post_ln: " << _is_post_ln << std::endl;
    std::cout << "image_size: " << _image_size << std::endl;
    std::cout << "patch_size: " << _patch_size << std::endl;
    std::cout << "channel_input: " << _channel_input << std::endl;
    std::cout << std::endl;
  }
};

}  // namespace cuda
}  // namespace lightseq
