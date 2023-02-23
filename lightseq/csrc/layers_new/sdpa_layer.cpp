#include "sdpa_layer.h"

namespace lightseq {

/*
Scaled Dot Product Attention
See paper "Attention is all you need" for details"
*/
template <typename T1, typename T2>
SDPALayer<T1, T2>::SDPALayer(int max_batch_tokens, int max_seq_len,
                             int head_dim, int num_heads,
                             float attn_prob_dropout_ratio)
    : Layer("SDPALayer"),
      // for training, max_batch_tokens =
      // max(batch_size * seq_len) for inference,
      // max_batch_tokens = max(batch_size *
      // beam_size * seq_len)
      _max_batch_tokens(max_batch_tokens),
      _max_seq_len(max_seq_len),
      _nhead(num_heads),
      _head_dim(head_dim) {
  float scale = float(1.0) / sqrt(float(_head_dim));
  _attn_scores = new StridedBatchGemmOp<T1, T2>(
      max_batch_tokens * num_heads * max_seq_len, scale, T1(0.0),
      MATRIX_OP::Transpose, MATRIX_OP::NonTranspose);
  _softmax = new SoftmaxOp<T1, T2>(max_batch_tokens, max_seq_len, num_heads);
  _attn_prob_dropout = new DropoutOp<T1, T2>(
      attn_prob_dropout_ratio, max_batch_tokens * num_heads * max_seq_len);
  _attn_context = new StridedBatchGemmOp<T1, T2>(
      max_batch_tokens * num_heads * head_dim, T1(1.0), T1(0.0),
      MATRIX_OP::NonTranspose, MATRIX_OP::NonTranspose);
  this->_context_ptr->exit_layer();  // necessary
}

template <typename T1, typename T2>
Variable* SDPALayer<T1, T2>::operator()(Variable* query, Variable* key,
                                        Variable* value, Variable* mask) {
  set_inputs({query, key, value, mask});

  Variable* attn_score = (*_attn_scores)(key, query);

  Variable* soft_out = (*_softmax)(attn_score, mask);
  Variable* attn_context = nullptr;
  if (_context_ptr->is_training()) {
    Variable* prob_dropout = (*_attn_prob_dropout)(soft_out);
    attn_context = (*_attn_context)(value, prob_dropout);
  } else {
    attn_context = (*_attn_context)(value, soft_out);
  }

  set_outputs({attn_context});
  return attn_context;
}

template <typename T1, typename T2>
void SDPALayer<T1, T2>::before_forward(int batch_size, int query_len,
                                       int kv_len, int kv_size,
                                       bool mask_future) {
  _softmax->before_forward(batch_size, query_len, kv_len, mask_future);

  _attn_prob_dropout->before_forward(batch_size * query_len * kv_len * _nhead);

  int n_batchgemm = batch_size * _nhead;
  _attn_scores->before_forward(kv_len, query_len, _head_dim, n_batchgemm,
                               kv_size);
  _attn_context->before_forward(_head_dim, query_len, kv_len, n_batchgemm,
                                kv_size);
}

template <typename T1, typename T2>
int SDPALayer<T1, T2>::load_para_and_grad(const T1* para_ptr,
                                          T2* grad_ptr) {  // for training
  return 0;
}

template <typename T1, typename T2>
int SDPALayer<T1, T2>::load_params(const std::vector<const T1*>& para_vec,
                                   int offset) {  // for inference
  return 0;
}

}  // namespace lightseq
