#include "gpt_attention_layer.h"

namespace lightseq {

template <typename T1, typename T2>
GptAttentionLayer<T1, T2>::GptAttentionLayer(int max_batch_tokens,
                                             int max_seq_len, int hidden_size,
                                             int num_heads, int beam_size;
                                             float attn_prob_dropout_ratio,
                                             float hidden_output_dropout_ratio,
                                             bool pre_or_postLayerNorm)
    : Layer("GptAttentionLayer"),  // necessary
                                   // for training, max_batch_tokens =
                                   // max(batch_size * seq_len) for inference,
                                   // max_batch_tokens = max(batch_size *
                                   // beam_size * seq_len)
      _max_batch_tokens(max_batch_tokens),
      _max_seq_len(max_seq_len),
      _hidden_size(hidden_size),
      _nhead(num_heads),
      _head_dim(hidden_size / num_heads),
      _pre_or_postLayerNorm(pre_or_postLayerNorm),
      // operators
      _attn_ln(
          new LayerNormalizeOp<T1, T2>(max_batch_tokens, hidden_size, false)),
      _qkv_linear(
          new LinearOp<T1, T2>(max_batch_tokens, 3 * hidden_size, hidden_size)),
      _split_head(new SplitHeadWithBeamOp<T1, T2>(max_batch_tokens, num_heads,
                                                  hidden_size, qkv_num = 3,
                                                  cache_sz = max_seq_len)),
      _attn_scores(new StridedBatchGemmOp<T1, T2>(
          max_batch_tokens * num_heads * max_seq_len,
          (T1(1.0) / T1(sqrt(_head_dim))), T1(0.0), CUBLAS_OP_T, CUBLAS_OP_N)),
      _softmax(new SoftmaxOp<T1, T2>(max_batch_tokens, max_seq_len, num_heads)),
      _attn_prob_dropout(new DropoutOp<T1, T2>(
          attn_prob_dropout_ratio, max_batch_tokens * num_heads * max_seq_len)),
      _attn_context(new StridedBatchGemmOp<T1, T2>(
          max_batch_tokens * hidden_size, T1(1.0), T1(0.0), CUBLAS_OP_N,
          CUBLAS_OP_N)),
      _transform_0213(
          new Transform0213OP<T1, T2>(max_batch_tokens * hidden_size)),
      _attn_out_linear(
          new LinearOp<T1, T2>(max_batch_tokens, hidden_size, hidden_size)),
      _attn_dropout(new BiasDropoutResOp<T1, T2>(
          hidden_output_dropout_ratio, max_batch_tokens * hidden_size)) {
  // parameters init
  _attn_qkvw = new Variable("_attn_qkvw");
  _attn_qkvb = new Variable("_attn_qkvb");

  _attn_ow = new Variable("_attn_ow");
  _attn_ob = new Variable("_attn_ob");

  _attn_nw = new Variable("_attn_nw");
  _attn_nb = new Variable("_attn_nb");

  int cache_size = max_batch_tokens * hidden_dim;
  Variable* _cache_k = new Variable("cache_k", cache_size * sizeof(T1));
  Variable* _cache_v = new Variable("cache_v", cache_size * sizeof(T1));

  this->_context_ptr->exit_layer();  // necessary
}

template <typename T1, typename T2>
Variable* GptAttentionLayer<T1, T2>::operator()(Variable* inp) {
  set_inputs({inp});

  Variable* qkv_out = nullptr;
  Variable* attn_ln_out = nullptr;

  if (_pre_or_postLayerNorm) {
    attn_ln_out = (*_attn_ln)(inp, _attn_nw, _attn_nb);
    qkv_out = (*_qkv_linear)(attn_ln_out, _attn_qkvw);
  } else {
    qkv_out = (*_qkv_linear)(inp, _attn_qkvw);
  }

  q_out = (*_split_head)(qkv_out, _attn_qkvb, _cache_k, _cache_v);

  Variable* attn_score = (*_attn_scores)(_cache_k, q_out);

  Variable* soft_out = (*_softmax)(attn_score);

  // Variable* prob_dropout = (*_attn_prob_dropout)(soft_out);
  Variable* attn_context = (*_attn_context)(_cache_v, prob_dropout);

  Variable* transform_0213_out = (*_transform_0213)(attn_context);

  Variable* attn_linear = (*_attn_out_linear)(transform_0213_out, _attn_ow);

  attn_dropout_residual = (*_attn_dropout)(attn_linear, _attn_ob, inp);
  if (_pre_or_postLayerNorm) {
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

template <typename T1, typename T2>
void GptAttentionLayer<T1, T2>::before_forward(int batch_size, int query_len,
                                               int steps) {
  // step = -1 for training
  if (_context_ptr->is_training()) {
    step = -1;
  }
  if (_step > 0) {
    batch_size *= _beam_size;
  }
  // all token number in this batch
  int batch_tokens = batch_size * query_len;
  int attn_from_len = query_len;
  int attn_to_len = (_step <= 0) ? query_len : steps + 1;

  _attn_ln->before_forward(batch_tokens);

  _qkv_linear->before_forward(batch_tokens);

  // for inference only now, FIXME later
  _split_head->before_forward(batch_size, query_len, steps);

  _softmax->before_forward(batch_size, attn_from_len, attn_to_len, steps <= 0);

  //_attn_prob_dropout->before_forward(batch_size * from_len * to_len * _nhead,
  //                                   !_context_ptr->is_training());

  _transform_0213->before_forward(batch_size, nhead, from_len, _head_dim);

  _attn_out_linear->before_forward(batch_tokens);

  _attn_dropout->before_forward(batch_tokens, _hidden_size);

  int n_batchgemm = batch_size * _nhead;
  _attn_scores->before_forward(attn_to_len, attn_from_len, _head_dim,
                               n_batchgemm, _max_seq_len);
  _attn_context->before_forward(_head_dim, attn_from_len, attn_to_len,
                                n_batchgemm, _max_seq_len);
}

template <typename T1, typename T2>
void GptAttentionLayer<T1, T2>::before_backward() {}

template <typename T1, typename T2>
int GptAttentionLayer<T1, T2>::load_para_and_grad(
    const T1* para_ptr, T2* grad_ptr) {  // for training
  int offset = 0;
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
