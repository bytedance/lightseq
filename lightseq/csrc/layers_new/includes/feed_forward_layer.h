#pragma once
#include "bias_act_dropout.h"
#include "bias_dropout_residual.h"
#include "linear.h"
#include "layer_normalize.h"
#include "layer.h"

namespace lightseq {

template <class T1, class T2>
class FeedForwardLayer : public Layer {
 private:
  // operators
  LayerNormalizeOp<T1, T2>* _ffn_ln = nullptr;
  LinearOp<T1, T2>* _ff1 = nullptr;
  BiasActDropoutOp<T1, T2>* _ffn_activation_dropout = nullptr;
  LinearOp<T1, T2>* _ff2 = nullptr;
  BiasDropoutResOp<T1, T2>* _ffn_dropout = nullptr;

  // parameters
  Variable* _inter_w;
  Variable* _inter_b;
  Variable* _output_w;
  Variable* _output_b;
  Variable* _ffn_nw;
  Variable* _ffn_nb;

  // shape related
  size_t _batch_dim;
  size_t _batch_heads;
  size_t _batch_tokens;

  size_t _layer_id;
  size_t _max_batch_tokens;
  size_t _max_seq_len;
  size_t _hidden_size;
  size_t _heads;
  size_t _intermediate_size;

  bool _pre_or_postLayerNorm;
  std::string _activation_fn;
  bool _is_post_ln;

 public:
  FeedForwardLayer(size_t layer_id, size_t max_batch_tokens, size_t max_seq_len,
                   size_t hidden_size, size_t num_heads, size_t intermediate_size,
                   float activation_dropout_ratio,
                   float hidden_output_dropout_ratio, bool pre_or_postLayerNorm,
                   std::string activation_fn, bool is_post_ln = false);

  virtual ~FeedForwardLayer() {}

  Variable* operator()(Variable* inp);

  void before_forward(int batch_size, int seq_len);

  void before_backward();

  int load_para_and_grad(const T1* para_ptr, T2* grad_ptr);

  int load_params(const std::vector<const T1*>& para_vec, int offset);
};

template class FeedForwardLayer<float, float>;
#ifdef LIGHTSEQ_cuda
template class FeedForwardLayer<__half, __half>;
#endif

template <class T1, class T2>
using FeedForwardLayerPtr = std::shared_ptr<FeedForwardLayer<T1, T2>>;

}  // namespace lightseq
