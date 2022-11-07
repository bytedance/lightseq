#pragma once
#include "layer.h"
#include "feed_forward_layer.h"
#include "dec_self_attention_layer.h"
#include "dec_enc_attention_layer.h"
#include "encdec_kv_layer.h"

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
  int _max_batch_tokens;
  int _hidden_size;
  int _step;
  int _batch_tokens;

  static Variable* total_enc_kv;

  Variable* enc_k;
  Variable* enc_v;

 public:
  TransformerDecoderLayer(int nshared_layer, int layer_id, int max_batch_tokens,
                          int _max_seq_len, int hidden_size, int num_heads,
                          int intermediate_size, float attn_dropout_ratio,
                          float hidden_output_dropout_ratio,
                          float activation_dropout_ratio,
                          bool pre_or_postLayerNorm, std::string activation_fn);

  virtual ~TransformerDecoderLayer();

  /*
    Inputs:
      index 0, Transformer encoder output;
      index 1,
  */
  std::tuple<Variable*, Variable*, Variable*> operator()(
      Variable* inp, Variable* enc_out, Variable* enc_mask,
      Variable* cache_self_k, Variable* cache_self_v);

  void before_forward(int batch_size, int trg_seq_len, int src_seq_len,
                      int step = -1);

  void before_backward() { return; }

  void forward_process() override;

  void backward_process() override;

  int load_para_and_grad(const T1* para_ptr, T2* grad_ptr);

  int load_params(const std::vector<const T1*>& para_vec, int offset, bool is_total_kv = false);
};

template class TransformerDecoderLayer<float, float>;
template class TransformerDecoderLayer<__half, __half>;

template <class T1, class T2>
using TransformerDecoderLayerPtr =
    std::shared_ptr<TransformerDecoderLayer<T1, T2>>;

}  // namespace lightseq
