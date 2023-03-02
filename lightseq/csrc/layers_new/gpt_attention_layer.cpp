#include "gpt_attention_layer.h"

namespace lightseq {

/*
for training,
max_batch_tokens = max(batch_size * seq_len)
for inference,
max_batch_tokens = max(batch_size * beam_size * seq_len)
*/
template <typename T1, typename T2>
GptAttentionLayer<T1, T2>::GptAttentionLayer(int max_batch_tokens,
                                             int max_seq_len, int hidden_size,
                                             int num_heads, int beam_size,
                                             float attn_prob_dropout_ratio,
                                             float hidden_output_dropout_ratio,
                                             bool is_pre_ln)
    : Layer("GptAttentionLayer"),
      _max_batch_tokens(max_batch_tokens),
      _max_seq_len(max_seq_len),
      _hidden_size(hidden_size),
      _nhead(num_heads),
      _head_dim(hidden_size / num_heads),
      _is_pre_ln(is_pre_ln) {
  // operators
  _attn_ln = new LayerNormalizeOp<T1, T2>(max_batch_tokens, hidden_size, false);
  _qkv_linear =
      new LinearOp<T1, T2>(max_batch_tokens, 3 * hidden_size, hidden_size);
  _split_head = new SplitHeadOp<T1, T2>(max_batch_tokens, num_heads,
                                        hidden_size, 3, max_seq_len);
  _sdpa = new SDPALayer<T1, T2>(max_batch_tokens, max_seq_len, _head_dim,
                                num_heads, 0.f);
  _transform_0213 = new Transform0213OP<T1, T2>(max_batch_tokens * hidden_size);
  _attn_out_linear =
      new LinearOp<T1, T2>(max_batch_tokens, hidden_size, hidden_size);
  _attn_dropout = new BiasDropoutResOp<T1, T2>(hidden_output_dropout_ratio,
                                               max_batch_tokens, hidden_size);
  // parameters init
  _attn_qkvw = new Variable("_attn_qkvw", g_dtype<T1>(), g_dtype<T2>());
  _attn_qkvb = new Variable("_attn_qkvb", g_dtype<T1>(), g_dtype<T2>());

  _attn_ow = new Variable("_attn_ow", g_dtype<T1>(), g_dtype<T2>());
  _attn_ob = new Variable("_attn_ob", g_dtype<T1>(), g_dtype<T2>());

  _attn_nw = new Variable("_attn_nw", g_dtype<T1>(), g_dtype<T2>());
  _attn_nb = new Variable("_attn_nb", g_dtype<T1>(), g_dtype<T2>());

  int cache_size = max_batch_tokens * hidden_size;
  Variable* _cache_k =
      new Variable("cache_k", cache_size, g_dtype<T1>(), g_dtype<T2>());

  Variable* _cache_v =
      new Variable("cache_v", cache_size, g_dtype<T1>(), g_dtype<T2>());

  this->_context_ptr->exit_layer();  // necessary
}

template <typename T1, typename T2>
Variable* GptAttentionLayer<T1, T2>::operator()(Variable* inp) {
  set_inputs({inp});

  Variable* qkv_out = nullptr;

  if (_is_pre_ln) {
    Variable* ln_res = (*_attn_ln)(inp, _attn_nw, _attn_nb);
    qkv_out = (*_qkv_linear)(ln_res, _attn_qkvw);
  } else {
    qkv_out = (*_qkv_linear)(inp, _attn_qkvw);
  }

  Variable* q_out = (*_split_head)(qkv_out, _attn_qkvb, _cache_k, _cache_v);

  // result of Scaled Dot Product Attention
  Variable* sdpa_res = (*_sdpa)(q_out, _cache_k, _cache_v);

  // [sz0, sz1, sz2, sz3] -> [sz0, sz2, sz1, sz3]
  Variable* transform_0213_out = (*_transform_0213)(sdpa_res);

  Variable* attn_linear = (*_attn_out_linear)(transform_0213_out, _attn_ow);

  Variable* attn_dropout_residual =
      (*_attn_dropout)(attn_linear, _attn_ob, inp);
  if (_is_pre_ln) {
    set_outputs({attn_dropout_residual});
    return attn_dropout_residual;
  }

  Variable* attn_ln_out =
      (*_attn_ln)(attn_dropout_residual, _attn_nw, _attn_nb);
  set_outputs({attn_ln_out});
  return attn_ln_out;
}

/*
template <typename T1, typename T2>
void GptAttentionLayer<T1, T2>::reorder_cache(Variable* input_ids) {
  xxx
}
*/

/*
for training or (inference and step=0)
batch_size = batch_size
for (inference and step>0)
batch_size = batch_size * beam_size
*/
template <typename T1, typename T2>
void GptAttentionLayer<T1, T2>::before_forward(int batch_size, int query_len,
                                               int steps) {
  // steps = -1 for training
  if (_context_ptr->is_training()) {
    steps = -1;
  }
  // all token number in this batch
  int batch_tokens = batch_size * query_len;
  int attn_from_len = query_len;
  int attn_to_len = (steps <= 0) ? query_len : steps + 1;

  _attn_ln->before_forward(batch_size, query_len);

  _qkv_linear->before_forward(batch_tokens);

  // for inference only now, FIXME later
  _split_head->before_forward(batch_size, query_len, steps);

  // mask future when training or (inference and steps=0)
  _sdpa->before_forward(batch_size, attn_from_len, attn_to_len, _max_seq_len,
                        steps <= 0);

  _transform_0213->before_forward(batch_size, _nhead, attn_from_len, _head_dim);

  _attn_out_linear->before_forward(batch_tokens);

  _attn_dropout->before_forward(batch_tokens, _hidden_size);
}

template <typename T1, typename T2>
void GptAttentionLayer<T1, T2>::before_backward() {}

template <typename T1, typename T2>
size_t GptAttentionLayer<T1, T2>::load_para_and_grad(
    const T1* para_ptr, T2* grad_ptr) {  // for training
  size_t offset = 0;
  _attn_qkvw->set_value((char*)(para_ptr + offset));
  _attn_qkvw->set_grad((char*)(grad_ptr + offset));
  offset += _hidden_size * _hidden_size * 3;

  _attn_qkvb->set_value((char*)(para_ptr + offset));
  _attn_qkvb->set_grad((char*)(grad_ptr + offset));
  offset += _hidden_size * 3;

  _attn_ow->set_value((char*)(para_ptr + offset));
  _attn_ow->set_grad((char*)(grad_ptr + offset));
  offset += _hidden_size * _hidden_size;

  _attn_ob->set_value((char*)(para_ptr + offset));
  _attn_ob->set_grad((char*)(grad_ptr + offset));
  offset += _hidden_size;

  _attn_nw->set_value((char*)(para_ptr + offset));
  _attn_nw->set_grad((char*)(grad_ptr + offset));
  offset += _hidden_size;

  _attn_nb->set_value((char*)(para_ptr + offset));
  _attn_nb->set_grad((char*)(grad_ptr + offset));
  offset += _hidden_size;

  return offset;
}

template <typename T1, typename T2>
int GptAttentionLayer<T1, T2>::load_params(
    const std::vector<const T1*>& para_vec, int offset) {  // for inference
  int size = 0;
  _attn_nw->set_value((char*)para_vec[offset + size]), size++;
  _attn_nb->set_value((char*)para_vec[offset + size]), size++;

  _attn_qkvw->set_value((char*)para_vec[offset + size]), size++;
  _attn_qkvb->set_value((char*)para_vec[offset + size]), size++;

  _attn_ow->set_value((char*)para_vec[offset + size]), size++;
  _attn_ob->set_value((char*)para_vec[offset + size]), size++;

  return size;
}

}  // namespace lightseq
