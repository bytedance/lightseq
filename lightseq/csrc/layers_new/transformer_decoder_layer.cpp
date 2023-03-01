#include "transformer_decoder_layer.h"

namespace lightseq {

template <typename T1, typename T2>
TransformerDecoderLayer<T1, T2>::TransformerDecoderLayer(
    int nshared_layer, int layer_id, int max_batch_tokens, int max_seq_len,
    int hidden_size, int num_heads, int intermediate_size,
    float attn_dropout_ratio, float hidden_output_dropout_ratio,
    float activation_dropout_ratio, bool is_pre_ln,
    std::string activation_fn, bool is_continuous_cache, int max_batch_size,
    int beam_size)
    : Layer("TransformerDecoderLayer"),
      _layer_id(layer_id),
      _nshared_layer(nshared_layer),
      _max_batch_tokens(max_batch_tokens),
      _hidden_size(hidden_size),
      _beam_size(beam_size) {
  int max_trg_tokens = _context_ptr->is_training() ? max_batch_tokens
                                                   : max_batch_size * beam_size;

  _self_attn_layer.reset(new DecSelfAttentionLayer<T1, T2>(
      layer_id, max_trg_tokens, max_seq_len, hidden_size, num_heads,
      attn_dropout_ratio, hidden_output_dropout_ratio, is_pre_ln,
      is_continuous_cache));

  _enc_attn_layer.reset(new DecEncAttentionLayer<T1, T2>(
      layer_id, max_trg_tokens, max_seq_len, hidden_size, num_heads,
      attn_dropout_ratio, hidden_output_dropout_ratio, is_pre_ln));

  _ffn_layer.reset(new FeedForwardLayer<T1, T2>(
      layer_id, max_trg_tokens, max_seq_len, hidden_size, num_heads,
      intermediate_size, activation_dropout_ratio, hidden_output_dropout_ratio,
      is_pre_ln, activation_fn));

  this->_context_ptr->exit_layer();  // necessary
}

template <typename T1, typename T2>
TransformerDecoderLayer<T1, T2>::~TransformerDecoderLayer() {}

template <typename T1, typename T2>
std::tuple<Variable*, Variable*, Variable*>
TransformerDecoderLayer<T1, T2>::operator()(Variable* inp,
                                            Variable* total_enc_kv,
                                            Variable* enc_mask,
                                            Variable* cache_self_k,
                                            Variable* cache_self_v) {
  set_inputs({inp, total_enc_kv, enc_mask, cache_self_k, cache_self_v});

  enc_k = new Variable("enc_k", total_enc_kv);
  enc_v = new Variable("enc_v", total_enc_kv);

  std::tuple<Variable*, Variable*, Variable*> self_attn_layer_product =
      (*_self_attn_layer)(inp, cache_self_k, cache_self_v);
  Variable* self_attn_out = std::get<0>(self_attn_layer_product);
  Variable* new_self_k = std::get<1>(self_attn_layer_product);
  Variable* new_self_v = std::get<2>(self_attn_layer_product);

  Variable* enc_attn_out =
      (*_enc_attn_layer)(self_attn_out, enc_mask, enc_k, enc_v);

  Variable* ffn_out = (*_ffn_layer)(enc_attn_out);

  set_outputs({ffn_out, new_self_k, new_self_v});

  return std::make_tuple(ffn_out, new_self_k, new_self_v);
}

template <typename T1, typename T2>
void TransformerDecoderLayer<T1, T2>::before_forward(
    size_t batch_size,
    size_t trg_seq_len,  // inference - beam_size;
    size_t src_seq_len, int step) {
  _step = step;
  _batch_size = batch_size;
  _batch_tokens = batch_size * trg_seq_len;

  enc_k->set_offset(2 * _layer_id * _hidden_size * batch_size * src_seq_len,
                    {batch_size, src_seq_len, _hidden_size});

  enc_v->set_offset(
      (2 * _layer_id + 1) * _hidden_size * batch_size * src_seq_len,
      {batch_size, src_seq_len, _hidden_size});

  _self_attn_layer->before_forward(batch_size, trg_seq_len, step);

  _enc_attn_layer->before_forward(batch_size, trg_seq_len, src_seq_len);

  _ffn_layer->before_forward(batch_size, trg_seq_len);
}

template <typename T1, typename T2>
size_t TransformerDecoderLayer<T1, T2>::load_para_and_grad(
    const T1* para_ptr, T2* grad_ptr) {  // for training

  size_t offset = 0;
  offset += _self_attn_layer->load_para_and_grad(para_ptr + offset,
                                                 grad_ptr + offset);
  offset +=
      _enc_attn_layer->load_para_and_grad(para_ptr + offset, grad_ptr + offset);
  offset +=
      _ffn_layer->load_para_and_grad(para_ptr + offset, grad_ptr + offset);
  offset += 24;  // for quant decoder

  return offset;
}

template <typename T1, typename T2>
int TransformerDecoderLayer<T1, T2>::load_params(
    const std::vector<const T1*>& para_vec, int offset) {  // for inference
  int size = 0;

  size += _self_attn_layer->load_params(para_vec, offset + size);
  size += _enc_attn_layer->load_params(para_vec, offset + size);
  size += _ffn_layer->load_params(para_vec, offset + size);

  return size;
}

}  // namespace lightseq
