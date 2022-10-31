#include "transformer_decoder_layer.h"

namespace lightseq {

template <typename T1, typename T2>
TransformerDecoderLayer<T1, T2>::TransformerDecoderLayer(
    int nshared_layer, int layer_id, int max_batch_tokens, int max_seq_len,
    int hidden_size, int num_heads, int intermediate_size,
    float attn_dropout_ratio, float hidden_output_dropout_ratio,
    float activation_dropout_ratio, bool pre_or_postLayerNorm,
    std::string activation_fn)
    : Layer("TransformerDecoderLayer"),
      _layer_id(layer_id),
      _nshared_layer(nshared_layer),
      _max_batch_tokens(max_batch_tokens),
      _hidden_size(hidden_size) {
  if (_layer_id == 0) {
    _enc_kv_layer.reset(new EncDecKvLayer<T1, T2>(
        nshared_layer, max_batch_tokens, hidden_size, num_heads));
  }

  _self_attn_layer.reset(new DecSelfAttentionLayer<T1, T2>(
      layer_id, max_batch_tokens, max_seq_len, hidden_size, num_heads,
      attn_dropout_ratio, hidden_output_dropout_ratio, pre_or_postLayerNorm));

  _enc_attn_layer.reset(new DecEncAttentionLayer<T1, T2>(
      layer_id, max_batch_tokens, max_seq_len, hidden_size, num_heads,
      attn_dropout_ratio, hidden_output_dropout_ratio, pre_or_postLayerNorm));

  _ffn_layer.reset(new FeedForwardLayer<T1, T2>(
      layer_id, max_batch_tokens, max_seq_len, hidden_size, num_heads,
      intermediate_size, activation_dropout_ratio, hidden_output_dropout_ratio,
      pre_or_postLayerNorm, activation_fn));

  this->_context_ptr->exit_layer();  // necessary
}

template <typename T1, typename T2>
TransformerDecoderLayer<T1, T2>::~TransformerDecoderLayer() {}

template <typename T1, typename T2>
std::tuple<Variable*, Variable*, Variable*>
TransformerDecoderLayer<T1, T2>::operator()(Variable* inp, Variable* enc_out,
                                            Variable* enc_mask,
                                            Variable* cache_self_k,
                                            Variable* cache_self_v) {
  LAYER_PRE_INPUTS({inp, enc_out, enc_mask, cache_self_k, cache_self_v});

  if (_layer_id == 0) {
    total_enc_kv = (*_enc_kv_layer)(enc_out);
  }

  enc_k = new Variable("enc_k", total_enc_kv);
  enc_v = new Variable("enc_v", total_enc_kv);

  std::tuple<Variable*, Variable*, Variable*> self_attn_layer_product =
      (*_self_attn_layer)(inp, cache_self_k, cache_self_v);
  Variable* self_attn_out = std::get<0>(self_attn_layer_product);
  Variable* new_self_k = std::get<1>(self_attn_layer_product);
  Variable* new_self_v = std::get<2>(self_attn_layer_product);

  Variable* enc_attn_out = (*_enc_attn_layer)(self_attn_out, enc_mask, enc_k, enc_v);

  Variable* ffn_out = (*_ffn_layer)(enc_attn_out);

  LAYER_POST_OUTPUTS({ffn_out, new_self_k, new_self_v});

  return std::make_tuple(ffn_out, new_self_k, new_self_v);
}

template <typename T1, typename T2>
void TransformerDecoderLayer<T1, T2>::forward() {
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
void TransformerDecoderLayer<T1, T2>::before_forward(int batch_size,
                                                     int trg_seq_len,
                                                     int src_seq_len,
                                                     int step) {
  _step = step;
  _batch_tokens = batch_size * trg_seq_len;

  enc_k->set_offset(2 * _layer_id * _hidden_size * batch_size * src_seq_len * sizeof(T1), 
                    2 * _layer_id * _hidden_size * batch_size * src_seq_len * sizeof(T2));

  enc_v->set_offset((2 * _layer_id + 1) * _hidden_size * batch_size * src_seq_len * sizeof(T1), 
                    (2 * _layer_id + 1) * _hidden_size * batch_size * src_seq_len * sizeof(T2));

  if (_layer_id == 0 && step <= 0) {
    _enc_kv_layer->before_forward(batch_size, src_seq_len);
  }

  _self_attn_layer->before_forward(batch_size, trg_seq_len, src_seq_len, step);

  _enc_attn_layer->before_forward(batch_size, trg_seq_len, src_seq_len);

  _ffn_layer->before_forward(batch_size, trg_seq_len);
}

template <typename T1, typename T2>
void TransformerDecoderLayer<T1, T2>::backward() {
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

  offset += _self_attn_layer->load_para_and_grad(para_ptr + offset,
                                                 grad_ptr + offset);

  offset +=
      _enc_attn_layer->load_para_and_grad(para_ptr + offset, grad_ptr + offset);

  offset +=
      _ffn_layer->load_para_and_grad(para_ptr + offset, grad_ptr + offset);

  offset += 24;  // for quant decoder

  if (_layer_id == 0) {
    offset +=
        _enc_kv_layer->load_para_and_grad(para_ptr + offset, grad_ptr + offset);
  }

  return offset;
}

template <typename T1, typename T2>
int TransformerDecoderLayer<T1, T2>::load_params(
    const std::vector<const T1*>& para_vec, int offset) {  // for inference
  int size = 0;

  return size;
}

template <typename T1, typename T2>
Variable* TransformerDecoderLayer<T1, T2>::total_enc_kv = nullptr;

}  // namespace lightseq
