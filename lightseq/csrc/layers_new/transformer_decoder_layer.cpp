#include "transformer_encoder_layer.h"

namespace lightseq {

template <typename T1, typename T2>
TransformerDecoderLayer<T1, T2>::TransformerDecoderLayer(
    int nshared_layer, int layer_id, int max_batch_tokens, int max_seq_len,
    int hidden_size, int num_heads, int intermediate_size,
    float attn_prob_dropout_ratio, float activation_dropout_ratio,
    float hidden_output_dropout_ratio, bool pre_or_postLayerNorm,
    std::string activation_fn, bool mask_future_tokens, bool is_post_ln)
    : Layer("TransformerDecoderLayer"),
      _layer_id(layer_id),
      _nshared_layer(nshared_layer) {
  if (_layer_id == 0) {
    _enc_kv_layer.reset(new EncDecKvLayer<T1, T2>(
        nshared_layer, layer_id, max_batch_tokens, hidden_size, num_heads));
  }

  _self_attn_layer.reset(new DecSelfAttentionLayer<T1, T2>(
      layer_id, max_batch_tokens, max_seq_len, hidden_size, num_heads,
      attn_prob_dropout_ratio, hidden_output_dropout_ratio,
      pre_or_postLayerNorm, mask_future_tokens, is_post_ln));

  _enc_attn_layer.reset(new DecEncAttentionLayer<T1, T2>(
      layer_id, max_batch_tokens, max_seq_len, hidden_size, num_heads,
      attn_prob_dropout_ratio, hidden_output_dropout_ratio,
      pre_or_postLayerNorm, mask_future_tokens, is_post_ln));

  _ffn_layer.reset(new FeedForwardLayer<T1, T2>(
      layer_id, max_batch_tokens, max_seq_len, hidden_size, num_heads,
      intermediate_size, activation_dropout_ratio, hidden_output_dropout_ratio,
      pre_or_postLayerNorm, activation_fn, is_post_ln));

  if (_encdec_kv_buffer == nullptr) {
    _encdec_kv_buffer =
        cuda_malloc<T1>(max_batch_tokens * 2 * hidden_size * nshared_layer);
    if (_context_ptr->is_training()) {
      _grad_encdec_kv_buffer =
          cuda_malloc<T2>(max_batch_tokens * 2 * hidden_size * nshared_layer);
    }
  }

  this->_context_ptr->exit_layer();  // necessary
}

template <typename T1, typename T2>
Variable* TransformerDecoderLayer<T1, T2>::operator()(Variable* inp,
                                                      Variable* input_mask,
                                                      Variable* enc_out,
                                                      Variable* cache_self_k,
                                                      Variable* cache_self_v) {
  LAYER_PRE_INPUTS({inp, input_mask, enc_out, cache_self_k, cache_self_v});

  Variable* enc_k;
  Variable* enc_v;

  if (_layer_id == 0) {
    std::tuple<Variable*, Variable*> enc_kv = (*_enc_kv_layer)(enc_out);
    enc_k = enc_kv.first;
    enc_v = enc_kv.second;
  } else {
    enc_k = new Variable("enc_k");
    enc_v = new Variable("enc_v");
  }

  enc_k->set_value(_encdec_kv_buffer);
  _encdec_kv_buffer += _nshared_layer * hidden_size * max_batch_tokens;
  enc_v->set_value(_encdec_kv_buffer);

  if (_context_ptr->is_training()) {
    enc_k->set_grad(_grad_encdec_kv_buffer);
    _grad_encdec_kv_buffer += _nshared_layer * hidden_size * max_batch_tokens;
    enc_v->set_grad(_grad_encdec_kv_buffer);
  }

  std::tuple<Variable*, Variable*, Variable*> self_attn_layer_product =
      (*_self_attn_layer)(inp, cache_self_k, cache_self_v);
  Variable* self_attn_out = std::get<0>(self_attn_layer_product);
  Variable* new_self_k = std::get<1>(self_attn_layer_product);
  Variable* new_self_v = std::get<2>(self_attn_layer_product);

  Variable* _enc_attn_out = (*_enc_attn_layer)(self_attn_out, enc_k, enc_v);

  Variable* ffn_out = (*_ffn_layer)(attn_out);

  LAYER_POST_OUTPUTS({ffn_out, new_self_k, new_self_v});
  return std::make_tuple(ffn_out, new_self_k, new_self_v);
}

template <typename T1, typename T2>
void forward() {
  if (_layer_id == 0 && _step <= 0) {
    _enc_kv_layer->forward();
  } else if (_layer_id == 0 && _step > 0) {
    _enc_kv_layer->tag_fw_flag();
  }
  _self_attn_layer->forward();
  _enc_attn_layer->forward();
  _ffn_layer->forward();
}

template <typename T1, typename T2>
void backward() {
  _ffn_layer->backward();
  _enc_attn_layer->backward();
  _self_attn_layer->backward();
  if (_layer_id == 0)  //  && _step == -1
    _enc_kv_layer->backward();
}

template <typename T1, typename T2>
int TransformerDecoderLayer<T1, T2>::load_para_and_grad(
    const T1* para_ptr, T2* grad_ptr) {  // for training
  int offset = 0;

  offset +=
      _attn_layer->load_para_and_grad(para_ptr + offset, grad_ptr + offset);

  offset +=
      _ffn_layer->load_para_and_grad(para_ptr + offset, grad_ptr + offset);

  return offset;
}

template <typename T1, typename T2>
int TransformerDecoderLayer<T1, T2>::load_params(
    const std::vector<const T1*>& para_vec, int offset) {  // for inference
  int size = 0;

  size += _attn_layer->load_params(para_vec, offset + size);

  size += _ffn_layer->load_params(para_vec, offset + size);

  return size;
}

}  // namespace lightseq
