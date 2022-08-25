#pragma once
#include "layer.h"
#include "feed_forward_layer.h"
#include "self_attention_layer.h"

namespace lightseq {

template <class T1, class T2>
class TransformerEncoderLayer : public Layer {
 private:
  SelfAttentionLayerPtr<T1, T2> _attn_layer;
  FeedForwardLayerPtr<T1, T2> _ffn_layer;

 public:
  TransformerEncoderLayer(int layer_id, int max_batch_tokens, int max_seq_len,
                          int hidden_size, int num_heads, int intermediate_size,
                          float attn_prob_dropout_ratio,
                          float activation_dropout_ratio,
                          float hidden_output_dropout_ratio,
                          bool pre_or_postLayerNorm, std::string activation_fn,
                          bool mask_future_tokens, const T1* para_ptr,
                          T2* grad_ptr, int& offset);
  virtual ~TransformerEncoderLayer() {}

  Variable* operator()(Variable* inp, Variable* inp_mask);

  void before_forward(int batch_size, int seq_len) {
    _attn_layer->before_forward(batch_size, seq_len);
    _ffn_layer->before_forward(batch_size, seq_len);
  }

  void before_backward() { return; }
};

template class TransformerEncoderLayer<float, float>;
template class TransformerEncoderLayer<__half, __half>;

template <class T1, class T2>
using TransformerEncoderLayerPtr =
    std::shared_ptr<TransformerEncoderLayer<T1, T2>>;

}  // namespace lightseq
