#include "feed_forward_layer.h"

namespace lightseq {

template <typename T1, typename T2>
FeedForwardLayer<T1, T2>::FeedForwardLayer(
    int layer_id, int max_batch_tokens, int max_seq_len, int hidden_size,
    int num_heads, int intermediate_size, float activation_dropout_ratio,
    float hidden_output_dropout_ratio, bool pre_or_postLayerNorm,
    std::string activation_fn, bool is_post_ln)
    : Layer("FeedForwardLayer"),
      _layer_id(layer_id),
      _max_batch_tokens(max_batch_tokens),
      _max_seq_len(max_seq_len),
      _hidden_size(hidden_size),
      _heads(num_heads),
      _intermediate_size(intermediate_size),
      _pre_or_postLayerNorm(pre_or_postLayerNorm),
      _activation_fn(activation_fn),
      _is_post_ln(is_post_ln),

      // operators
      _ffn_ln(new LayerNormalizeOp<T1, T2>(max_batch_tokens, hidden_size)),
      _ff1(new LinearOp<T1, T2>(max_batch_tokens, intermediate_size,
                                hidden_size)),
      _ffn_activation_dropout(new BiasActDropoutOp<T1, T2>(
          activation_dropout_ratio, max_batch_tokens * intermediate_size,
          activation_fn)),
      _ff2(new LinearOp<T1, T2>(max_batch_tokens, hidden_size,
                                intermediate_size)),
      _ffn_dropout(new BiasDropoutResOp<T1, T2>(
          hidden_output_dropout_ratio, max_batch_tokens * hidden_size)) {
  // parameters node
  _inter_w = new Variable("_inter_w");
  _inter_b = new Variable("_inter_b");

  _output_w = new Variable("_output_w");
  _output_b = new Variable("_output_b");

  _ffn_nw = new Variable("_ffn_nw");
  _ffn_nb = new Variable("_ffn_nb");

  this->_context_ptr->exit_layer();  // necessary
}

template <typename T1, typename T2>
Variable* FeedForwardLayer<T1, T2>::operator()(Variable* inp) {
  set_inputs({inp});
  Variable* ff1_out = nullptr;
  Variable* ffn_ln_out = nullptr;
  if (_pre_or_postLayerNorm) {
    ffn_ln_out = (*_ffn_ln)(inp, _ffn_nw, _ffn_nb);
    ff1_out = (*_ff1)(ffn_ln_out, _inter_w);
  } else {
    ff1_out = (*_ff1)(inp, _inter_w);
  }

  Variable* ffn_act_out = (*_ffn_activation_dropout)(ff1_out, _inter_b);

  Variable* ff2_out = (*_ff2)(ffn_act_out, _output_w);

  Variable* ffn_dropout_residual;
  if (_pre_or_postLayerNorm && _is_post_ln) {
    ffn_dropout_residual = (*_ffn_dropout)(ff2_out, _output_b, ffn_ln_out);
  } else {
    ffn_dropout_residual = (*_ffn_dropout)(ff2_out, _output_b, inp);
  }

  if (!_pre_or_postLayerNorm) {
    Variable* ffn_ln_out = (*_ffn_ln)(ffn_dropout_residual, _ffn_nw, _ffn_nb);
    set_outputs({ffn_ln_out});
    return ffn_ln_out;
  } else {
    set_outputs({ffn_dropout_residual});
    return ffn_dropout_residual;
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
void FeedForwardLayer<T1, T2>::before_backward() {}

template <typename T1, typename T2>
int FeedForwardLayer<T1, T2>::load_para_and_grad(
    const T1* para_ptr,
    T2* grad_ptr) {  // for training
  int offset = 0;

  _inter_w->set_value((char*)(para_ptr + offset));
  _inter_w->set_grad((char*)(grad_ptr + offset));
  offset += _hidden_size * _intermediate_size;

  _inter_b->set_value((char*)(para_ptr + offset));
  _inter_b->set_grad((char*)(grad_ptr + offset));
  offset += _intermediate_size;

  _output_w->set_value((char*)(para_ptr + offset));
  _output_w->set_grad((char*)(grad_ptr + offset));
  offset += _hidden_size * _intermediate_size;

  _output_b->set_value((char*)(para_ptr + offset));
  _output_b->set_grad((char*)(grad_ptr + offset));
  offset += _hidden_size;

  _ffn_nw->set_value((char*)(para_ptr + offset));
  _ffn_nw->set_grad((char*)(grad_ptr + offset));
  offset += _hidden_size;

  _ffn_nb->set_value((char*)(para_ptr + offset));
  _ffn_nb->set_grad((char*)(grad_ptr + offset));
  offset += _hidden_size;

  return offset;
}

template <typename T1, typename T2>
int FeedForwardLayer<T1, T2>::load_params(
    const std::vector<const T1*>& para_vec,
    int offset) {  // for inference
  int size = 0;
  _ffn_nw->set_value((char*)para_vec[offset + size]), size++;
  _ffn_nb->set_value((char*)para_vec[offset + size]), size++;

  _inter_w->set_value((char*)para_vec[offset + size]), size++;
  _inter_b->set_value((char*)para_vec[offset + size]), size++;

  _output_w->set_value((char*)para_vec[offset + size]), size++;
  _output_b->set_value((char*)para_vec[offset + size]), size++;

  return size;
}

}  // namespace lightseq
