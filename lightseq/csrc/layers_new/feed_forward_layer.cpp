#include "feed_forward_layer.h"

namespace lightseq {

template <typename T1, typename T2>
FeedForwardLayer<T1, T2>::FeedForwardLayer(
    size_t layer_id, size_t max_batch_tokens, size_t max_seq_len,
    size_t hidden_size, size_t num_heads, size_t intermediate_size,
    float activation_dropout_ratio, float hidden_output_dropout_ratio,
    bool is_pre_ln, std::string activation_fn)
    : Layer("FeedForwardLayer"),
      _layer_id(layer_id),
      _max_batch_tokens(max_batch_tokens),
      _max_seq_len(max_seq_len),
      _hidden_size(hidden_size),
      _heads(num_heads),
      _intermediate_size(intermediate_size),
      _is_pre_ln(is_pre_ln),
      _activation_fn(activation_fn),

      // operators
      _ffn_ln(new LayerNormalizeOp<T1, T2>(max_batch_tokens, hidden_size)),
      _ff1(new LinearOp<T1, T2>(max_batch_tokens, intermediate_size,
                                hidden_size)),
      _ffn_activation_dropout(new BiasActDropoutOp<T1, T2>(
          activation_dropout_ratio, max_batch_tokens, intermediate_size,
          activation_fn)),
      _ff2(new LinearOp<T1, T2>(max_batch_tokens, hidden_size,
                                intermediate_size)),
      _ffn_dropout(new BiasDropoutResOp<T1, T2>(
          hidden_output_dropout_ratio, max_batch_tokens, hidden_size)) {
  // parameters node
  _inter_w = new Variable("_inter_w", g_dtype<T1>(), g_dtype<T2>());
  _inter_b = new Variable("_inter_b", g_dtype<T1>(), g_dtype<T2>());

  _output_w = new Variable("_output_w", g_dtype<T1>(), g_dtype<T2>());
  _output_b = new Variable("_output_b", g_dtype<T1>(), g_dtype<T2>());

  _ffn_nw = new Variable("_ffn_nw", g_dtype<T1>(), g_dtype<T2>());
  _ffn_nb = new Variable("_ffn_nb", g_dtype<T1>(), g_dtype<T2>());

  this->_context_ptr->exit_layer();  // necessary
}

template <typename T1, typename T2>
Variable* FeedForwardLayer<T1, T2>::operator()(Variable* inp) {
  set_inputs({inp});
  Variable* ff1_out = nullptr;
  if (_is_pre_ln) {
    Variable* ffn_ln_out = (*_ffn_ln)(inp, _ffn_nw, _ffn_nb);
    ff1_out = (*_ff1)(ffn_ln_out, _inter_w);
  } else {
    ff1_out = (*_ff1)(inp, _inter_w);
  }

  Variable* ffn_act_out = (*_ffn_activation_dropout)(ff1_out, _inter_b);

  Variable* ff2_out = (*_ff2)(ffn_act_out, _output_w);

  Variable* ffn_dropout_residual = (*_ffn_dropout)(ff2_out, _output_b, inp);
  if (_is_pre_ln) {
    set_outputs({ffn_dropout_residual});
    return ffn_dropout_residual;
  }

  Variable* ffn_ln_out = (*_ffn_ln)(ffn_dropout_residual, _ffn_nw, _ffn_nb);
  set_outputs({ffn_ln_out});
  return ffn_ln_out;
}

template <typename T1, typename T2>
void FeedForwardLayer<T1, T2>::before_forward(int batch_size, int seq_len) {
  int batch_tokens = batch_size * seq_len;

  _ffn_ln->before_forward(batch_size, seq_len);

  _ff1->before_forward(batch_tokens);

  _ffn_activation_dropout->before_forward(batch_tokens, _intermediate_size);

  _ff2->before_forward(batch_tokens);

  _ffn_dropout->before_forward(batch_tokens, _hidden_size);
}

template <typename T1, typename T2>
void FeedForwardLayer<T1, T2>::before_backward() {}

template <typename T1, typename T2>
size_t FeedForwardLayer<T1, T2>::load_para_and_grad(
    const T1* para_ptr,
    T2* grad_ptr) {  // for training
  size_t offset = 0;

  _inter_w->set_value((char*)(para_ptr + offset));
  _inter_w->set_grad((char*)(grad_ptr + offset));
  _inter_w->set_shape({_intermediate_size, _hidden_size});
  offset += _hidden_size * _intermediate_size;

  _inter_b->set_value((char*)(para_ptr + offset));
  _inter_b->set_grad((char*)(grad_ptr + offset));
  _inter_b->set_shape({_intermediate_size});
  offset += _intermediate_size;

  _output_w->set_value((char*)(para_ptr + offset));
  _output_w->set_grad((char*)(grad_ptr + offset));
  _output_w->set_shape({_hidden_size, _intermediate_size});
  offset += _hidden_size * _intermediate_size;

  _output_b->set_value((char*)(para_ptr + offset));
  _output_b->set_grad((char*)(grad_ptr + offset));
  _output_b->set_shape({_hidden_size});
  offset += _hidden_size;

  _ffn_nw->set_value((char*)(para_ptr + offset));
  _ffn_nw->set_grad((char*)(grad_ptr + offset));
  _ffn_nw->set_shape({_hidden_size});
  offset += _hidden_size;

  _ffn_nb->set_value((char*)(para_ptr + offset));
  _ffn_nb->set_grad((char*)(grad_ptr + offset));
  _ffn_nb->set_shape({_hidden_size});
  offset += _hidden_size;

  return offset;
}

template <typename T1, typename T2>
int FeedForwardLayer<T1, T2>::load_params(
    const std::vector<const T1*>& para_vec,
    int offset) {  // for inference
  int size = 0;
  _ffn_nw->set_value((char*)para_vec[offset + size]), size++;
  _ffn_nw->set_shape({_hidden_size});
  _ffn_nb->set_value((char*)para_vec[offset + size]), size++;
  _ffn_nb->set_shape({_hidden_size});

  _inter_w->set_value((char*)para_vec[offset + size]), size++;
  _inter_w->set_shape({_intermediate_size, _hidden_size});
  _inter_b->set_value((char*)para_vec[offset + size]), size++;
  _inter_b->set_shape({_intermediate_size});

  _output_w->set_value((char*)para_vec[offset + size]), size++;
  _output_w->set_shape({_hidden_size, _intermediate_size});
  _output_b->set_value((char*)para_vec[offset + size]), size++;
  _output_b->set_shape({_hidden_size});

  return size;
}

}  // namespace lightseq
