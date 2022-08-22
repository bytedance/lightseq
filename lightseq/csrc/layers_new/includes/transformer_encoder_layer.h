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
                          bool mask_future_tokens);
  virtual ~TransformerEncoderLayer() {}

  Variable* operator()(Variable* inp, Variable* inp_mask) { return output; }

  void before_forward(int size) {
    // op before forward
    _operator_add->before_forward(size);
    _operator_add2->before_forward(size);
  }

  void before_backward() { return; }
};

template class TransformerEncoderLayer<float, float>;
template class TransformerEncoderLayer<__half, __half>;

template <class T1, class T2>
using TransformerEncoderLayerPtr =
    std::shared_ptr<TransformerEncoderLayer<T1, T2>>;

}  // namespace lightseq
