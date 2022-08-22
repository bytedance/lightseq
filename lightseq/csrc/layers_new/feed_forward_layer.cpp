#include "feed_forward_layer.h"

namespace lightseq {

template <typename T1, typename T2>
FeedForwardLayer<T1, T2>::FeedForwardLayer(
    int layer_id, int max_batch_tokens, int max_seq_len, int hidden_size,
    int num_heads, int intermediate_size, float activation_dropout_ratio,
    bool pre_or_postLayerNorm, std::string activation_fn, const T1* para_ptr,
    T2* grad_ptr, int& offset)
    : _layer_id(layer_id),
      _max_batch_tokens(max_batch_tokens),
      _max_seq_len(max_seq_len),
      _max_batchs(max_batch_tokens / max_seq_len),
      _hidden_size(hidden_size),
      _heads(num_heads),
      _intermediate_size(intermediate_size),
      _pre_or_postLayerNorm(pre_or_postLayerNorm),
      _activation_fn(activation_fn),

      // operators
      _ffn_ln(new NormalizeLayerOp<T1, T2>(_max_batch_tokens, _hidden_size)),
      _ff1(new FeedForwardOp<T1, T2>(_max_batch_tokens, intermediate_size,
                                     hidden_size)),
      _ffn_activation_dropout(new BiasActDropoutOp<T1, T2>(
          activation_dropout_ratio, _max_batch_tokens * intermediate_size,
          activation_fn)),
      _ff2(new FeedForwardOp<T1, T2>(_max_batch_tokens, hidden_size,
                                     intermediate_size)),
      _ffn_dropout(new BiasDropoutResOp<T1, T2>(
          hidden_output_dropout_ratio, _max_batch_tokens * hidden_size)) {
  // parameters node
  _inter_w = new Variable(this->_name + "_inter_w", (char*)(para_ptr + offset),
                          (char*)(grad_ptr + offset));
  offset += _hidden_size * _intermediate_size;
  _inter_b = new Variable(this->_name + "_inter_b", (char*)(para_ptr + offset),
                          (char*)(grad_ptr + offset));
  offset += _intermediate_size;

  _output_w =
      new Variable(this->_name + "_output_w", (char*)(para_ptr + offset),
                   (char*)(grad_ptr + offset));
  offset += _hidden_size * _intermediate_size;
  _output_b =
      new Variable(this->_name + "_output_b", (char*)(para_ptr + offset),
                   (char*)(grad_ptr + offset));
  offset += _hidden_size;

  _ffn_nw = new Variable(this->_name + "_ffn_nw", (char*)(para_ptr + offset),
                         (char*)(grad_ptr + offset));
  offset += _hidden_size;
  _ffn_nb = new Variable(this->_name + "_ffn_nb", (char*)(para_ptr + offset),
                         (char*)(grad_ptr + offset));
  offset += _hidden_size;

  this->_context_ptr->exit_layer();  // necessary
}

template <typename T1, typename T2>
Variable* FeedForwardLayer<T1, T2>::operator()(Variable* inp) {
  Variable* ff1_out = nullptr;
  if (_pre_or_postLayerNorm) {
    Variable* ffn_ln_out = (*_ffn_ln)(inp, _ffn_nw, _ffn_nb);
    ff1_out = (*_ff1)(ffn_ln_out, _inter_w);
  } else {
    ff1_out = (*_ff1)(inp, _inter_w);
  }

  Variable* ffn_act_out = (*_ffn_activation_dropout)(ff1_out, _inter_b);

  Variable* ff2_out = (*_ff2)(ffn_act_out, _output_w);

  Variable* ffn_dropout = (*_ffn_dropout)(ff2_out, _output_b, inp);

  if (!_pre_or_postLayerNorm) {
    Variable* ffn_ln_out = (*_ffn_ln)(inp, _ffn_nw, _ffn_nb);
    return ffn_ln_out;
  } else {
    return ffn_dropout;
  }
}

template <typename T1, typename T2>
void FeedForwardLayer<T1, T2>::before_forward(int batch_size, int seq_len) {
  int batch_tokens = batch_size * seq_len;

  _ffn_ln->before_forward(batch_tokens);

  _ff1->before_forward(batch_tokens);

  _ffn_activation_dropout->before_forward(batch_tokens, _intermediate_size);

  _ff2->before_forward(batch_tokens);

  _ffn_dropout->before_forward(batch_tokens, _hidden_size);
}

template <typename T1, typename T2>
void FeedForward<T1, T2>::before_backward() {}

}  // namespace lightseq
