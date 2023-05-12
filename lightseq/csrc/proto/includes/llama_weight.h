#pragma once
#include "proto_headers.h"
#include "proto_util.h"
#include "hdf5_util.h"
#include "model_base.h"

namespace lightseq {

/*
Load the model weights which stored in custom proto file into GPU memory.
*/
template <typename T>
class LlamaWeight {
 private:
  cudaStream_t stream;
  GenerateConfig* _gen_conf;
  T float2required(float value);

  // parsing function for hdf5
  void hdf5_get_model_config(hid_t hdf5_file);
  void hdf5_parse_emb_wei(hid_t hdf5_file);
  void hdf5_parse_enc_wei(hid_t hdf5_file);

  // store the weights pointer
  std::vector<const T *> _p_d_src_emb_wei;  // size: 4
  std::vector<const T *> _p_d_enc_wei;      // size: 12 * enc_layer_num

  // store the weights on gpu memory
  std::vector<T *> _d_src_emb_wei;
  std::vector<T *> _d_enc_wei;

 public:
  std::string initializing(std::string weight_path, GenerateConfig* gen_conf);

  const std::vector<const T *> &get_src_emb_wei() const {
    // {token_emb, pos_emb, norm_scale, norm_bias}
    return _p_d_src_emb_wei;
  }

  const std::vector<const T *> &get_enc_wei() const {
    // {multihead_norm_scale, multihead_norm_bias, multihead_qkv_kernel,
    // multihead_qkv_bias multihead_output_kernel, multihead_output_bias
    // ffn_norm_scale, ffn_norm_bias}
    // ffn_first_kernel, ffn_first_bias, ffn_second_kernel, ffn_second_bias} *
    // encoder_layer_num
    return _p_d_enc_wei;
  }

  size_t _hidden_size;
  int _inner_size;
  int _max_step;
  int _extra_decode_length;
  int _src_vocab_size;
  int _layer_num;  // number of encoder layer
  int _dim_per_head;
  int _weight_per_enc_layer;  // 12

  int _head_num;
  int _padding_id;  // for src
  std::string _generate_method = "topk";
  int _topk = 1;
  float _topp = 0.75;
  int _eos_id;

  int _beam_size = 1;
  float _length_penalty = 1.0;
  float _diverse_lambda = 0.;
  bool _use_gelu = true;

  void print_model_config() {
    std::cout << "***model config***" << std::endl;
    std::cout << "decoder layers: " << _layer_num << std::endl;
    std::cout << "hidden size: " << _hidden_size << std::endl;
    std::cout << "inner size: " << _inner_size << std::endl;
    std::cout << "head number: " << _head_num << std::endl;
    std::cout << "dim per head: " << _dim_per_head << std::endl;
    std::cout << "src vocab size: " << _src_vocab_size << std::endl;
    std::cout << std::endl;
    std::cout << "***generator config***" << std::endl;
    std::cout << "generate method: " << _generate_method << std::endl;
    std::cout << "beam size: " << _beam_size << std::endl;
    std::cout << "max step: " << _max_step << std::endl;
    std::cout << "length penalty: " << _length_penalty << std::endl;
    std::cout << "diverse lambda: " << _diverse_lambda << std::endl;
    std::cout << std::endl;
    _gen_conf->print_config();
  }
};

}  // namespace lightseq
