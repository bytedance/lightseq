#pragma once
#include "bias_act_dropout.h"
#include "bias_dropout_residual.h"
#include "feed_forward.h"
#include "normalize_layer.h"
#include "layer.h"

namespace lightseq {

class FeedForwardLayerWeight {
 public:
  FeedForwardLayerWeight(int hidden_size, int intermediate_size)
      : _hidden_size(hidden_size), _intermediate_size(intermediate_size) {}
  char* _inter_w_ptr;
  char* _inter_b_ptr;
  char* _output_w_ptr;
  char* _output_b_ptr;
  char* _ffn_nw_ptr;
  char* _ffn_nb_ptr;

  char* _grad_inter_w_ptr;
  char* _grad_inter_b_ptr;
  char* _grad_output_w_ptr;
  char* _grad_output_b_ptr;
  char* _grad_ffn_nw_ptr;
  char* _grad_ffn_nb_ptr;

  int _hidden_size;
  int _intermediate_size;

  template <class T1, class T2>
  int load_para_and_grad(const T1* para_ptr, T2* grad_ptr);

  template <typename T>
  int load_params(const std::vector<const T*>& para_vec);
};

using FeedForwardLayerWeightPtr = std::shared_ptr<FeedForwardLayerWeight>;

template <class T1, class T2>
class FeedForwardLayer : public Layer {
 private:
  // operators
  NormalizeLayerOp<T1, T2>* _ffn_ln = nullptr;
  FeedForwardOp<T1, T2>* _ff1 = nullptr;
  BiasActDropoutOp<T1, T2>* _ffn_activation_dropout = nullptr;
  FeedForwardOp<T1, T2>* _ff2 = nullptr;
  BiasDropoutResOp<T1, T2>* _ffn_dropout = nullptr;

  // parameters
  Variable* _inter_w;
  Variable* _inter_b;
  Variable* _output_w;
  Variable* _output_b;
  Variable* _ffn_nw;
  Variable* _ffn_nb;

  // shape related
  int _batch_dim;
  int _batch_heads;
  int _batch_tokens;

  int _layer_id;
  int _max_batch_tokens;
  int _max_seq_len;
  int _hidden_size;
  int _heads;
  int _intermediate_size;
  bool _pre_or_postLayerNorm;
  std::string _activation_fn;

 public:
  FeedForwardLayer(int layer_id, int max_batch_tokens, int max_seq_len,
                   int hidden_size, int num_heads, int intermediate_size,
                   float activation_dropout_ratio,
                   float hidden_output_dropout_ratio, bool pre_or_postLayerNorm,
                   std::string activation_fn, FeedForwardLayerWeightPtr ffn_wt);

  virtual ~FeedForwardLayer() {}

  Variable* operator()(Variable* inp);

  void before_forward(int batch_size, int seq_len);

  void before_backward();
};

template class FeedForwardLayer<__half, __half>;
template class FeedForwardLayer<float, float>;

template <class T1, class T2>
using FeedForwardLayerPtr = std::shared_ptr<FeedForwardLayer<T1, T2>>;

}  // namespace lightseq
