#include "self_attention_layer.h"

namespace lightseq {

template <typename T1, typename T2>
SelfAttentionLayer<T1, T2>::SelfAttentionLayer(
    int layer_id, int max_batch_tokens, 
    int max_seq_len, int hidden_size,
    int num_heads, float attn_prob_dropout_ratio,
    float activation_dropout_ratio, 
    float hidden_output_dropout_ratio,
    bool pre_or_postLayerNorm, bool mask_future_tokens, 
    const T1* para_ptr, T2* grad_ptr, int &offset): 
      Layer("SelfAttentionLayer"),
      _layer_id(layer_id),
      _max_batch_tokens(max_batch_tokens),
      _max_batch(max_batch_tokens / max_seq_len),
      _max_seq_len(max_seq_len),
      _hidden_size(hidden_size),
      _heads(num_heads),
      _training(true),
      _pre_or_postLayerNorm(pre_or_postLayerNorm),
      // operators
      _attn_ln(new NormalizeLayerOp<T1, T2>(max_batch_tokens, hidden_size, false)),
      _qkv_linear(new FeedForwardOp<T1, T2>(max_batch_tokens, 3 * hidden_size, hidden_size)), 
      _bias_add_transform_20314(new BiasAddTrans20314<T1, T2>(_max_batch, _max_seq, _heads, hidden_size)),
      _attn_scores(new StridedBatchGemmOp<T1, T2>(_max_batch_tokens * _heads * _max_seq_len, (T1(1.0) / T1(sqrt(_hidden_size / _heads))), T1(0.0), CUBLAS_OP_T, CUBLAS_OP_N)),
      _softmax(new SoftmaxOp<T1, T2>(_max_batch, _max_seq_len, _max_seq_len, _heads, mask_future_tokens)),
      _attn_prob_dropout(new DropoutOp<T1, T2>(attn_prob_dropout_ratio, _max_batch_tokens * _heads * _max_seq_len)),
      _attn_context(new StridedBatchGemmOp<T1, T2>(_max_batch_tokens * _hidden_size, T1(1.0), T1(0.0), CUBLAS_OP_N, CUBLAS_OP_N))),
      _transform_0213(new Transform0213<T1, T2>(_max_batch, _max_seq_len, _heads, hidden_size)),
      _attn_out_linear(new FeedForwardOp<T1, T2>(max_batch_tokens, hidden_size, hidden_size)),
      _attn_dropout(new BiasDropoutResOp<T1, T2>(hidden_output_dropout_ratio, _max_batch_tokens * _hidden_size)) {
    // parameters
    _attn_qkvw = new Variable(this->_name + "_attn_qkvw", para_ptr + offset, grad_ptr + offset);
    offset += _hidden_size * _hidden_size * 3;
    _attn_qkvb = new Variable(this->_name + "_attn_qkvb", para_ptr + offset, grad_ptr + offset);
    offset += _hidden_size * 3;

    _attn_ow = new Variable(this->_name + "_attn_ow", para_ptr + offset, grad_ptr + offset);
    offset += _hidden_size * _hidden_size;
    _attn_ob = new Variable(this->_name + "_attn_ob", para_ptr + offset, grad_ptr + offset);
    offset += _hidden_size;
    
    _attn_nw = new Variable(this->_name + "_attn_nw", para_ptr + offset, grad_ptr + offset);
    offset += _hidden_size;
    _attn_nb = new Variable(this->_name + "_attn_nb", para_ptr + offset, grad_ptr + offset);
    offset += _hidden_size;
    
    this->_context_ptr->exit_layer();  // necessary
}

template <typename T1, typename T2>
void SelfAttentionLayer<T1, T2>::operator()(Variable* inp, Variable* inp_mask) {
    Variable* qkv_out = nullptr;
    
    if (_pre_or_postLayerNorm) {
        Variable* attn_ln_out = (*_attn_ln)(inp, _attn_nw, _attn_nb);
        qkv_out = (*_qkv_linear)(attn_ln_out, _attn_qkvw);        
    }
    else {
        qkv_out = (*_qkv_linear)(inp, _attn_qkvw);
    }
    
    std::tuple<Variable*, Variable*, Variable*> transform_20314_out = (*_bias_add_transform_20314)(qkv_out, _attn_qkvb);
    Variable* q_out = std::get<0>(transform_20314_out);
    Variable* k_out = std::get<1>(transform_20314_out);
    Variable* v_out = std::get<2>(transform_20314_out);
    
    Variable* attn_score = (*_attn_scores)(k_out, q_out);
    
    Variable* soft_out = (*_softmax)(attn_score, inp_mask);
    
    Variable* prob_dropout = (*_attn_prob_dropout)(soft_out);
    
    Variable* attn_context = (*_attn_context)(v_out, prob_dropout);
    
    Variable* transform_0213_out = (*_transform_0213)(attn_context);

    Variable* attn_linear = (*_attn_out_linear)(transform_0213_out, _attn_ow);

    Variable* attn_dropout = (*_attn_dropout)(attn_linear, _attn_ob);

    if (!_pre_or_postLayerNorm) {
        Variable* attn_ln_out = (*_attn_ln)(attn_dropout, _attn_nw, _attn_nb);
        return attn_ln_out;
    }
    else {
        return attn_dropout;
    }
}

template <typename T1, typename T2>
void SelfAttentionLayer<T1, T2>::before_forward(int batch_size, int seq_len) {
    _batch_tokens = batch_size * seq_len;
    _batch_heads = batch_size * _heads;
    _batch_dim = _batch_tokens * _hidden_size;

    _attn_ln->before_forward(_batch_tokens);

    _qkv_linear->before_forward(_batch_tokens);

    _bias_add_transform_20314->before_forward(batch_size, seq_len);

    _attn_scores->before_forward(seq_len, seq_len, _hidden_size / _heads, _batch_heads);

    _softmax->before_forward(batch_size, seq_len, seq_len);

    _attn_prob_dropout->before_forward(_batch_heads * seq_len * seq_len);

    _attn_context->before_forward(_hidden_size / _heads, seq_len, seq_len, _batch_heads);

    _transform_0213->before_forward(batch_size, seq_len);

    _attn_out_linear->before_forward(_batch_tokens);

    _attn_dropout->before_forward(_batch_tokens, _hidden_size);
}

template <typename T1, typename T2>
void SelfAttentionLayer<T1, T2>::before_backward() {

}

}