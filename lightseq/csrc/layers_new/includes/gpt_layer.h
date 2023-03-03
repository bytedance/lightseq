#pragma once
#include "layer.h"
#include "feed_forward_layer.h"
#include "gpt_attention_layer.h"

namespace lightseq {

template <class T1, class T2>
class GptLayer : public Layer {
 private:
  GptAttentionLayerPtr<T1, T2> _attn_layer;
  FeedForwardLayerPtr<T1, T2> _ffn_layer;

  int _layer_id;

 public:
  GptLayer(int layer_id, int max_batch_tokens, int max_seq_len, int hidden_size,
           int num_heads, int intermediate_size, float attn_prob_dropout_ratio,
           float activation_dropout_ratio, float hidden_output_dropout_ratio,
           std::string activation_fn, bool mask_future_tokens,
           int beam_size = 1);
  virtual ~GptLayer() {}

  Variable* operator()(Variable* inp);

  void before_forward(int batch_size, int seq_len, int steps) {
    _attn_layer->before_forward(batch_size, seq_len, steps);
    _ffn_layer->before_forward(batch_size, seq_len);
  }

  size_t load_para_and_grad(const T1* para_ptr, T2* grad_ptr);

  int load_params(const std::vector<const T1*>& para_vec, int offset);
};

template class GptLayer<float, float>;
#ifdef LIGHTSEQ_cuda
template class GptLayer<__half, __half>;
#endif

template <class T1, class T2>
using GptLayerPtr = std::shared_ptr<GptLayer<T1, T2>>;

}  // namespace lightseq
