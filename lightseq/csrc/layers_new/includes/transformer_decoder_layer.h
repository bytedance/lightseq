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

  size_t _layer_id;
  size_t _batch_size;
  size_t _nshared_layer;
  size_t _max_batch_tokens;
  size_t _hidden_size;
  size_t _batch_tokens;
  size_t _beam_size;

  int _step;

  static Variable* total_enc_kv;

  Variable* enc_k;
  Variable* enc_v;

 public:
  TransformerDecoderLayer(int nshared_layer, int layer_id, int max_batch_tokens,
                          int _max_seq_len, int hidden_size, int num_heads,
                          int intermediate_size, float attn_dropout_ratio,
                          float hidden_output_dropout_ratio,
                          float activation_dropout_ratio,
                          bool pre_or_postLayerNorm, std::string activation_fn,
                          bool is_post_ln = false,
                          bool is_continuous_cache = true,
                          int max_batch_size = 1, int beam_size = 1);

  virtual ~TransformerDecoderLayer();

  /*
    Inputs:
      index 0, Transformer encoder output;
      index 1, cache_k_new;
      index 2, cache_v_new;
  */
  std::tuple<Variable*, Variable*, Variable*> operator()(
      Variable* inp, Variable* enc_out, Variable* enc_mask,
      Variable* cache_self_k, Variable* cache_self_v);

  void before_forward(size_t batch_size, size_t trg_seq_len, size_t src_seq_len,
                      int step = -1);

  void before_backward() { return; }

  void forward_process() override;

  void backward_process() override;

  int load_para_and_grad(const T1* para_ptr, T2* grad_ptr);

  int load_params(const std::vector<const T1*>& para_vec, int offset,
                  bool is_total_kv = false);
};

template class TransformerDecoderLayer<float, float>;
#ifdef LIGHTSEQ_cuda
template class TransformerDecoderLayer<__half, __half>;
#endif

template <class T1, class T2>
using TransformerDecoderLayerPtr =
    std::shared_ptr<TransformerDecoderLayer<T1, T2>>;

template <class T1, class T2>
class TransformerDecoderLayerV2 : public Layer {
 private:
  DecSelfAttentionLayerPtr<T1, T2> _self_attn_layer;
  DecEncAttentionLayerPtr<T1, T2> _enc_attn_layer;
  FeedForwardLayerPtr<T1, T2> _ffn_layer;

  size_t _layer_id;
  size_t _batch_size;
  size_t _nshared_layer;
  size_t _max_batch_tokens;
  size_t _hidden_size;
  size_t _batch_tokens;
  size_t _beam_size;

  int _step;

  static Variable* total_enc_kv;

  Variable* enc_k;
  Variable* enc_v;

 public:
  TransformerDecoderLayerV2(int nshared_layer, int layer_id,
                            int max_batch_tokens, int _max_seq_len,
                            int hidden_size, int num_heads,
                            int intermediate_size, float attn_dropout_ratio,
                            float hidden_output_dropout_ratio,
                            float activation_dropout_ratio,
                            bool pre_or_postLayerNorm,
                            std::string activation_fn, bool is_post_ln = false,
                            bool is_continuous_cache = true,
                            int max_batch_size = 1, int beam_size = 1);

  virtual ~TransformerDecoderLayerV2();

  /*
    Inputs:
      index 0, Transformer encoder output;
      index 1, cache_k_new;
      index 2, cache_v_new;
  */
  std::tuple<Variable*, Variable*, Variable*> operator()(
      Variable* inp, Variable* totoal_enc_kv, Variable* enc_mask,
      Variable* cache_self_k, Variable* cache_self_v);

  void before_forward(size_t batch_size, size_t trg_seq_len, size_t src_seq_len,
                      int step = -1);

  void before_backward() { return; }

  int load_para_and_grad(const T1* para_ptr, T2* grad_ptr);

  int load_params(const std::vector<const T1*>& para_vec, int offset);
};

template class TransformerDecoderLayerV2<float, float>;
#ifdef LIGHTSEQ_cuda
template class TransformerDecoderLayerV2<__half, __half>;
#endif

template <class T1, class T2>
using TransformerDecoderLayerV2Ptr =
    std::shared_ptr<TransformerDecoderLayerV2<T1, T2>>;

}  // namespace lightseq
