#include "dec_self_attention_layer.h"

namespace lightseq {

template <typename T1, typename T2>
DecSelfAttentionLayer<T1, T2>::DecSelfAttentionLayer(
    int layer_id, int max_batch_tokens, int max_seq_len, int hidden_size,
    int num_heads, float attn_prob_dropout_ratio,
    float hidden_output_dropout_ratio, bool pre_or_postLayerNorm,
    bool mask_future_tokens, bool is_post_ln)
    : Layer("DecSelfAttentionLayer"),  // necessary
      _layer_id(layer_id),
      _max_batch_tokens(max_batch_tokens),
      _max_seq_len(max_seq_len),
      _hidden_size(hidden_size),
      _heads(num_heads),
      _pre_or_postLayerNorm(pre_or_postLayerNorm),
      _is_post_ln(is_post_ln),
      // operators
      _attn_ln(
          new LayerNormalizeOp<T1, T2>(max_batch_tokens, hidden_size, false)),
      _qkv_linear(new FeedForwardOp<T1, T2>(max_batch_tokens, 3 * hidden_size,
                                            hidden_size)),
      _bias_add_transform_20314(new BiasAddTrans20314<T1, T2>(
          max_batch_tokens, num_heads, hidden_size, 3)),
      _deal_cache_k(new LaunchConcat3Dim1<T1, T2>(num_heads, hidden_size)),
      _deal_cache_v(new LaunchConcat3Dim1<T1, T2>(num_heads, hidden_size)),
      _attn_scores(new StridedBatchGemmOp<T1, T2>(
          max_batch_tokens * num_heads * max_seq_len,
          (T1(1.0) / T1(sqrt(hidden_size / num_heads))), T1(0.0), CUBLAS_OP_T,
          CUBLAS_OP_N)),
      _softmax(new SoftmaxOp<T1, T2>(max_batch_tokens, max_seq_len, num_heads,
                                     mask_future_tokens)),
      _attn_prob_dropout(new DropoutOp<T1, T2>(
          attn_prob_dropout_ratio, max_batch_tokens * num_heads * max_seq_len)),
      _attn_context(new StridedBatchGemmOp<T1, T2>(
          max_batch_tokens * hidden_size, T1(1.0), T1(0.0), CUBLAS_OP_N,
          CUBLAS_OP_N)),
      _transform_0213(
          new Transform0213<T1, T2>(max_batch_tokens, num_heads, hidden_size)),
      _attn_out_linear(new FeedForwardOp<T1, T2>(max_batch_tokens, hidden_size,
                                                 hidden_size)),
      _attn_dropout(new BiasDropoutResOp<T1, T2>(
          hidden_output_dropout_ratio, max_batch_tokens * hidden_size)) {
  // parameters
  _attn_qkvw = new Variable("_attn_qkvw");
  _attn_qkvb = new Variable("_attn_qkvb");

  _attn_ow = new Variable("_attn_ow");
  _attn_ob = new Variable("_attn_ob");

  _attn_nw = new Variable("_attn_nw");
  _attn_nb = new Variable("_attn_nb");

  this->_context_ptr->exit_layer();  // necessary
}

template <typename T1, typename T2>
std::tuple<Variable*, Variable*, Variable*>
DecSelfAttentionLayer<T1, T2>::operator()(Variable* inp, Variable* cache_k,
                                          Variable* cache_v) {
  LAYER_PRE_INPUTS({inp, cache_k, cache_v});
  
  Variable* qkv_out = nullptr;
  Variable* attn_ln_out = nullptr;

  if (_pre_or_postLayerNorm) {
    attn_ln_out = (*_attn_ln)(inp, _attn_nw, _attn_nb);
    qkv_out = (*_qkv_linear)(attn_ln_out, _attn_qkvw);
  } else {
    qkv_out = (*_qkv_linear)(inp, _attn_qkvw);
  }

  std::tuple<Variable*, Variable*, Variable*> transform_20314_out =
      (*_bias_add_transform_20314)(qkv_out, _attn_qkvb);
  Variable* q_out = std::get<0>(transform_20314_out);
  Variable* k_out = std::get<1>(transform_20314_out);
  Variable* v_out = std::get<2>(transform_20314_out);

  Variable* new_k_out = (*_deal_cache_k)(k_out, cache_k);
  Variable* new_v_out = (*_deal_cache_v)(v_out, cache_v);

  Variable* attn_score = (*_attn_scores)(new_k_out, q_out);

  Variable* soft_out = (*_softmax)(attn_score);

  Variable* attn_context = nullptr;
  if (_context_ptr->is_training()) {
    Variable* prob_dropout = (*_attn_prob_dropout)(soft_out);
    attn_context = (*_attn_context)(new_v_out, prob_dropout);
  } else {
    attn_context = (*_attn_context)(new_v_out, soft_out);
  }

  Variable* transform_0213_out = (*_transform_0213)(attn_context);

  Variable* attn_linear = (*_attn_out_linear)(transform_0213_out, _attn_ow);

  Variable* attn_dropout_residual;
  if (_pre_or_postLayerNorm && _is_post_ln) {
    attn_dropout_residual =
        (*_attn_dropout)(attn_linear, _attn_ob, attn_ln_out);
  } else {
    attn_dropout_residual = (*_attn_dropout)(attn_linear, _attn_ob, inp);
  }

  if (!_pre_or_postLayerNorm) {
    Variable* attn_ln_out =
        (*_attn_ln)(attn_dropout_residual, _attn_nw, _attn_nb);
    LAYER_POST_OUTPUTS({attn_ln_out, new_k_out, new_v_out});
    return std::make_tuple(attn_ln_out, new_k_out, new_v_out);
  } else {
    LAYER_POST_OUTPUTS({attn_dropout_residual, new_k_out, new_v_out});
    return std::make_tuple(attn_dropout_residual, new_k_out, new_v_out);
  }
}

template <typename T1, typename T2>
void DecSelfAttentionLayer<T1, T2>::before_forward(int batch_size, int trg_seq_len, int src_seq_len,
                                                   int steps, bool predict) {
  _batch_tokens = batch_size * seq_len;
  _batch_heads = batch_size * _heads;
  _batch_dim = _batch_tokens * _hidden_size;

  _attn_ln->before_forward(_batch_tokens);

  _qkv_linear->before_forward(_batch_tokens);

  _bias_add_transform_20314->before_forward(batch_size, seq_len);

  _deal_cache_k->before_forward(batch_size, seq_len, steps, predict);

  _deal_cache_v->before_forward(batch_size, seq_len, steps, predict);

  _attn_scores->before_forward(seq_len, seq_len, _hidden_size / _heads,
                               _batch_heads);

  _softmax->before_forward(batch_size, seq_len, seq_len);

  _attn_prob_dropout->before_forward(_batch_heads * seq_len * seq_len);

  _attn_context->before_forward(_hidden_size / _heads, seq_len, seq_len,
                                _batch_heads);

  _transform_0213->before_forward(batch_size, seq_len);

  _attn_out_linear->before_forward(_batch_tokens);

  _attn_dropout->before_forward(_batch_tokens, _hidden_size);
}

template <typename T1, typename T2>
void DecSelfAttentionLayer<T1, T2>::before_backward() {}

template <typename T1, typename T2>
int DecSelfAttentionLayer<T1, T2>::load_para_and_grad(
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
int DecSelfAttentionLayer<T1, T2>::load_params(
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
