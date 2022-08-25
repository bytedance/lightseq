#pragma once
#include "layer.h"
#include "feed_forward_layer.h"
#include "multihead_attention_layer.h"

namespace lightseq {

class TransformerEncoderLayerWeight {
 private:
  int _hidden_size;
  int _intermediate_size;

 public:
  TransformerEncoderLayerWeight(int hidden_size, int intermediate_size)
      : _hidden_size(hidden_size),
        _intermediate_size(intermediate_size),
        _ffn_layer_wt(
            new FeedForwardLayerWeight(hidden_size, intermediate_size)),
        _attn_layer_wt(new MultiheadAttentionLayerWeight(hidden_size)) {}

  FeedForwardLayerWeightPtr _ffn_layer_wt;

  MultiheadAttentionLayerWeightPtr _attn_layer_wt;

  template <class T1, class T2>
  int load_para_and_grad(const T1* para_ptr, T2* grad_ptr);

  template <typename T>
  int load_params(const std::vector<const T*>& para_vec);
};

using TransformerEncoderLayerWeightPtr =
    std::shared_ptr<TransformerEncoderLayerWeight>;

template <class T1, class T2>
class TransformerEncoderLayer : public Layer {
 private:
  MultiheadAttentionLayerPtr<T1, T2> _attn_layer;
  FeedForwardLayerPtr<T1, T2> _ffn_layer;

 public:
  TransformerEncoderLayer(int layer_id, int max_batch_tokens, int max_seq_len,
                          int hidden_size, int num_heads, int intermediate_size,
                          float attn_prob_dropout_ratio,
                          float activation_dropout_ratio,
                          float hidden_output_dropout_ratio,
                          bool pre_or_postLayerNorm, std::string activation_fn,
                          bool mask_future_tokens,
                          TransformerEncoderLayerWeightPtr enc_layer_wt);
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
