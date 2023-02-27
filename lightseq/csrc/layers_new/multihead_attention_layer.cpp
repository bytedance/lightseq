#include "multihead_attention_layer.h"

namespace lightseq {

template <typename T1, typename T2>
MultiheadAttentionLayer<T1, T2>::MultiheadAttentionLayer(
    int layer_id, int max_batch_tokens, int max_seq_len, int hidden_size,
    int num_heads, float attn_prob_dropout_ratio,
    float hidden_output_dropout_ratio, bool pre_or_postLayerNorm,
    bool mask_future_tokens)
    : Layer("MultiheadAttentionLayer"),  // necessary
      _layer_id(layer_id),
      _max_batch_tokens(max_batch_tokens),
      _max_seq_len(max_seq_len),
      _hidden_size(hidden_size),
      _heads(num_heads),
      _pre_or_postLayerNorm(pre_or_postLayerNorm),
      // operators
      _attn_ln(
          new LayerNormalizeOp<T1, T2>(max_batch_tokens, hidden_size, false)),
      _qkv_linear(
          new LinearOp<T1, T2>(max_batch_tokens, 3 * hidden_size, hidden_size)),
      _bias_add_transform_20314(new BiasAddTrans20314<T1, T2>(
          max_batch_tokens, num_heads, hidden_size, 3)),

      _sdpa_layer(new SDPALayer<T1, T2>(max_batch_tokens, max_seq_len,
                                        hidden_size / num_heads, num_heads,
                                        attn_prob_dropout_ratio)),

      _transform_0213(
          new Transform0213OP<T1, T2>(max_batch_tokens * hidden_size)),
      _attn_out_linear(
          new LinearOp<T1, T2>(max_batch_tokens, hidden_size, hidden_size)),
      _attn_dropout(new BiasDropoutResOp<T1, T2>(
          hidden_output_dropout_ratio, max_batch_tokens, hidden_size)) {
  // parameters
  _attn_qkvw = new Variable("_attn_qkvw", g_dtype<T1>(), g_dtype<T2>());
  _attn_qkvb = new Variable("_attn_qkvb", g_dtype<T1>(), g_dtype<T2>());

  _attn_ow = new Variable("_attn_ow", g_dtype<T1>(), g_dtype<T2>());
  _attn_ob = new Variable("_attn_ob", g_dtype<T1>(), g_dtype<T2>());

  _attn_nw = new Variable("_attn_nw", g_dtype<T1>(), g_dtype<T2>());
  _attn_nb = new Variable("_attn_nb", g_dtype<T1>(), g_dtype<T2>());

  this->_context_ptr->exit_layer();  // necessary
}

template <typename T1, typename T2>
Variable* MultiheadAttentionLayer<T1, T2>::operator()(Variable* inp,
                                                      Variable* inp_mask) {
  Variable* qkv_out = nullptr;
  Variable* attn_ln_out = nullptr;
  set_inputs({inp, inp_mask});
  if (_pre_or_postLayerNorm) {
    attn_ln_out = (*_attn_ln)(inp, _attn_nw, _attn_nb);
    qkv_out = (*_qkv_linear)(attn_ln_out, _attn_qkvw);
  } else {
    qkv_out = (*_qkv_linear)(inp, _attn_qkvw);
  }

  Variable* transform_20314_out =
      (*_bias_add_transform_20314)(qkv_out, _attn_qkvb);
  q_out = new Variable("q_out", transform_20314_out);
  k_out = new Variable("k_out", transform_20314_out);
  v_out = new Variable("v_out", transform_20314_out);

  Variable* sdpa_out = (*_sdpa_layer)(q_out, k_out, v_out, inp_mask);

  Variable* transform_0213_out = (*_transform_0213)(sdpa_out);

  Variable* attn_linear = (*_attn_out_linear)(transform_0213_out, _attn_ow);

  Variable* attn_dropout_residual =
      (*_attn_dropout)(attn_linear, _attn_ob, inp);
  if (_pre_or_postLayerNorm) {
    set_outputs({attn_dropout_residual});
    return attn_dropout_residual;
  }

  Variable* post_ln = (*_attn_ln)(attn_dropout_residual, _attn_nw, _attn_nb);
  set_outputs({post_ln});
  return post_ln;
}

template <typename T1, typename T2>
void MultiheadAttentionLayer<T1, T2>::before_forward(size_t batch_size,
                                                     size_t seq_len) {
  _batch_tokens = batch_size * seq_len;
  _batch_heads = batch_size * _heads;
  _batch_dim = _batch_tokens * _hidden_size;

  _attn_ln->before_forward(batch_size, seq_len);

  _qkv_linear->before_forward(_batch_tokens);

  _bias_add_transform_20314->before_forward(batch_size, seq_len);

  q_out->set_offset(0, {batch_size, _heads, seq_len, _hidden_size / _heads});
  k_out->set_offset(_batch_dim,
                    {batch_size, _heads, seq_len, _hidden_size / _heads});
  v_out->set_offset(2 * _batch_dim,
                    {batch_size, _heads, seq_len, _hidden_size / _heads});

  _sdpa_layer->before_forward(batch_size, seq_len, seq_len, -1, false);

  _transform_0213->before_forward(batch_size, _heads, seq_len,
                                  _hidden_size / _heads);

  _attn_out_linear->before_forward(_batch_tokens);

  _attn_dropout->before_forward(_batch_tokens, _hidden_size);
}

template <typename T1, typename T2>
size_t MultiheadAttentionLayer<T1, T2>::load_para_and_grad(
    const T1* para_ptr, T2* grad_ptr) {  // for training
  size_t offset = 0;
  _attn_qkvw->set_value((char*)(para_ptr + offset));
  _attn_qkvw->set_grad((char*)(grad_ptr + offset));
  _attn_qkvw->set_shape({3, _hidden_size, _hidden_size});
  offset += _hidden_size * _hidden_size * 3;

  _attn_qkvb->set_value((char*)(para_ptr + offset));
  _attn_qkvb->set_grad((char*)(grad_ptr + offset));
  _attn_qkvb->set_shape({3, _hidden_size});
  offset += _hidden_size * 3;

  _attn_ow->set_value((char*)(para_ptr + offset));
  _attn_ow->set_grad((char*)(grad_ptr + offset));
  _attn_ow->set_shape({_hidden_size, _hidden_size});
  offset += _hidden_size * _hidden_size;

  _attn_ob->set_value((char*)(para_ptr + offset));
  _attn_ob->set_grad((char*)(grad_ptr + offset));
  _attn_ob->set_shape({_hidden_size});
  offset += _hidden_size;

  _attn_nw->set_value((char*)(para_ptr + offset));
  _attn_nw->set_grad((char*)(grad_ptr + offset));
  _attn_nw->set_shape({_hidden_size});
  offset += _hidden_size;

  _attn_nb->set_value((char*)(para_ptr + offset));
  _attn_nb->set_grad((char*)(grad_ptr + offset));
  _attn_nb->set_shape({_hidden_size});
  offset += _hidden_size;

  return offset;
}

template <typename T1, typename T2>
int MultiheadAttentionLayer<T1, T2>::load_params(
    const std::vector<const T1*>& para_vec, int offset) {  // for inference
  int size = 0;
  _attn_nw->set_value((char*)para_vec[offset + size]), size++;
  _attn_nb->set_value((char*)para_vec[offset + size]), size++;

  _attn_qkvw->set_value((char*)para_vec[offset + size]), size++;
  _attn_qkvw->set_shape({3, _hidden_size, _hidden_size});
  _attn_qkvb->set_value((char*)para_vec[offset + size]), size++;
  _attn_qkvb->set_shape({3, _hidden_size});

  _attn_ow->set_value((char*)para_vec[offset + size]), size++;
  _attn_ow->set_shape({_hidden_size});
  _attn_ob->set_value((char*)para_vec[offset + size]), size++;
  _attn_ob->set_shape({_hidden_size});

  return size;
}

}  // namespace lightseq
