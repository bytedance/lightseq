#include "transformer_encoder_layer.h"

namespace lightseq {

template <typename T1, typename T2>
TransformerDecoderLayer<T1, T2>::TransformerDecoderLayer(
    int nshared_layer, int layer_id, int max_batch_tokens, int max_seq_len, int hidden_size,
    int num_heads, int intermediate_size, float attn_prob_dropout_ratio,
    float activation_dropout_ratio, float hidden_output_dropout_ratio,
    bool pre_or_postLayerNorm, std::string activation_fn,
    bool mask_future_tokens, bool is_post_ln)
    : Layer("TransformerDecoderLayer"),
      _layer_id(layer_id),
      _nshared_layer(nshared_layer) {

  if(_layer_id == 0){
    _enc_kv_layer.reset(new EncDecKvLayer<T1, T2>(nshared_layer, layer_id, max_batch_tokens, hidden_size, num_heads));
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

  this->_context_ptr->exit_layer();  // necessary
}

template <typename T1, typename T2>
Variable* TransformerDecoderLayer<T1, T2>::operator()(Variable* inp, Variable* enc_out, Variable* cache_self_k, 
                                                      Variable* cache_self_v) {

  std::tuple<Variable*, Variable*> enc_kv;
  if(_layer_id == 0){
    enc_kv = (*_enc_kv_layer)(enc_out);
    // regist at _context_ptr
  }
  else {
    enc_kv = std::make_tuple(enc_k, enc_v);
  }

  std::tuple<Variable*, Variable*, Variable*> _self_attn_out = (*_self_attn_layer)(inp, cache_self_k, cache_self_v);

  Variable* _enc_attn_out = (*_enc_attn_layer)(std::get<0>(_self_attn_out), std::get<0>(enc_kv), std::get<1>(enc_kv));

  Variable* ffn_out = (*_ffn_layer)(attn_out);

  return std::make_tuple(ffn_out, );
}

template <typename T1, typename T2>
void forward() {
  if (_layer_id == 0 && _step <= 0) {
    _enc_kv_layer->forward();
  }
  else if(_layer_id == 0 && _step > 0) {
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
  if(_layer_id == 0) //  && _step == -1
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
