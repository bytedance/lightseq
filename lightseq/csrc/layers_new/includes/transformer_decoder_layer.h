#pragma once
#include "layer.h"
#include "feed_forward_layer.h"
#include "dec_self_attention_layer.h"

namespace lightseq {

template <class T1, class T2>
class TransformerDecoderLayer : public Layer {
 private:
  EncDecKvLayerPtr<T1, T2> _enc_kv_layer;
  DecSelfAttentionLayerPtr<T1, T2> _self_attn_layer;
  DecEncAttentionLayerPtr<T1, T2> _enc_attn_layer;
  FeedForwardLayerPtr<T1, T2> _ffn_layer;

  int _layer_id;
  int _nshared_layer;

 public:
  TransformerDecoderLayer(int nshared_layer, int layer_id, int max_batch_tokens, int max_seq_len,
                          int hidden_size, int num_heads, int intermediate_size,
                          float attn_prob_dropout_ratio,
                          float activation_dropout_ratio,
                          float hidden_output_dropout_ratio,
                          bool pre_or_postLayerNorm, std::string activation_fn,
                          bool mask_future_tokens, bool is_post_ln = false);
                          
  virtual ~TransformerDecoderLayer() {}

  /* 
    Inputs: 
      index 0, Transformer encoder output;
      index 1, 
  */
  Variable* operator()(Variable* inp, Variable* input_mask, 
                      Variable* enc_out, Variable* cache_self_k, 
                      Variable* cache_self_v);

  void before_forward(int batch_size, int trg_seq_len, int src_seq_len, int step = -1) {

     if (step >= 0) {
      _predict = true;
      _attn_scores.SetConfig(step + 1, 1, _hidden_size / _heads);
      _attn_context.SetConfig(_hidden_size / _heads, 1, step + 1);
    } else {
      _predict = false;
      _attn_scores.SetConfig(_trg_seq_len, _trg_seq_len, _hidden_size / _heads);
      _attn_context.SetConfig(_hidden_size / _heads, _trg_seq_len,
                              _trg_seq_len);
    }

    if(_layer_id == 0 && step <= 0){
      _enc_kv_layer->before_forward(batch_size, seq_len);
    }
    _self_attn_layer->before_forward(batch_size, trg_seq_len, src_seq_len, step);
    _enc_attn_layer->before_forward(batch_size, trg_seq_len, src_seq_len);
    _ffn_layer->before_forward(batch_size, seq_len);
  }

  void before_backward() { return; }

  void forward() override;

  void backward() override;

  int load_para_and_grad(const T1* para_ptr, T2* grad_ptr);

  int load_params(const std::vector<const T1*>& para_vec, int offset);
};

template class TransformerDecoderLayer<float, float>;
template class TransformerDecoderLayer<__half, __half>;

template <class T1, class T2>
using TransformerDecoderLayerPtr =
    std::shared_ptr<TransformerDecoderLayer<T1, T2>>;

}  // namespace lightseq
