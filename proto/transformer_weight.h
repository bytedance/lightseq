#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <thrust/device_vector.h>

#include "src/custom/transformer/proto/transformer.pb.h"

namespace lab {
namespace nmt {
/*
    extract the Transformer network weights from proto to gpu device memory
*/
class TransformerWeight {
 public:
  std::string initializing(std::string proto_path);
  const std::vector<const float*>& get_src_emb_wei() const;
  const std::vector<const float*>& get_trg_emb_wei() const;
  const std::vector<const float*>& get_enc_wei() const;
  const std::vector<const float*>& get_dec_wei() const;

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

 private:
  void get_model_config(const Transformer& transformer);
  std::string parse_emb_wei(const EmbeddingLayer& layer, std::string source);
  std::string parse_enc_wei(const Transformer& transformer);
  std::string parse_dec_wei(const Transformer& transformer);

  std::vector<const float*> _p_d_src_emb_wei;  // size: 4
  std::vector<const float*> _p_d_trg_emb_wei;  // size: 4
  std::vector<const float*> _p_d_enc_wei;      // size: 12 * enc_layer_num
  std::vector<const float*> _p_d_dec_wei;      // size: 18 * dec_layer_num

  // store the weights on gpu memo
  thrust::device_vector<float> _d_src_emb_wei;
  thrust::device_vector<float> _d_trg_emb_wei;
  thrust::device_vector<float> _d_enc_wei;
  thrust::device_vector<float> _d_dec_wei;
};

}  // namespace nmt
}  // namespace lab
