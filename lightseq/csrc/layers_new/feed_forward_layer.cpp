#include "feed_forward_layer.h"

namespace lightseq {

template<class T1, class T2>
int FeedForwardLayerWeight::load_para_and_grad(const T1* para_ptr, T2* grad_ptr)  { // for training
  int offset = 0;
  _inter_w_ptr = (char*)(para_ptr + offset);
  _grad_inter_w_ptr = (char*)(grad_ptr + offset);
  offset += _hidden_size * _intermediate_size;

  _inter_b_ptr = (char*)(para_ptr + offset);
  _grad_inter_b_ptr = (char*)(grad_ptr + offset);
  offset += _intermediate_size;

  _output_w_ptr = (char*)(para_ptr + offset);
  _grad_output_w_ptr = (char*)(grad_ptr + offset);
  offset += _hidden_size * _intermediate_size;

  _output_b_ptr = (char*)(para_ptr + offset);
  _grad_output_b_ptr = (char*)(grad_ptr + offset);
  offset += _hidden_size;

  _ffn_nw_ptr = (char*)(para_ptr + offset);
  _grad_ffn_nw_ptr = (char*)(grad_ptr + offset);
  offset += _hidden_size;

  _ffn_nb_ptr = (char*)(para_ptr + offset);
  _grad_ffn_nb_ptr = (char*)(grad_ptr + offset);
  offset += _hidden_size;

  return offset;
}

template int FeedForwardLayerWeight::load_para_and_grad(const float* para_ptr, float* grad_ptr);
template int FeedForwardLayerWeight::load_para_and_grad(const __half* para_ptr, __half* grad_ptr);

template<typename T>
int FeedForwardLayerWeight::load_params(const std::vector<const T*> & para_vec) { // for inference
  int offset = 0;
  _ffn_nw_ptr = (char*)para_vec[offset ++];
  _ffn_nb_ptr = (char*)para_vec[offset ++];

  _inter_w_ptr = (char*)para_vec[offset ++];
  _inter_b_ptr = (char*)para_vec[offset ++];

  _output_w_ptr = (char*)para_vec[offset ++];
  _output_b_ptr = (char*)para_vec[offset ++];

  return offset;
}

template int FeedForwardLayerWeight::load_params<float>(const std::vector<const float*> & para_vec);
template int FeedForwardLayerWeight::load_params<__half>(const std::vector<const __half*> & para_vec);

template <typename T1, typename T2>
FeedForwardLayer<T1, T2>::FeedForwardLayer(
    int layer_id, int max_batch_tokens, int max_seq_len, int hidden_size,
    int num_heads, int intermediate_size, float activation_dropout_ratio,
    float hidden_output_dropout_ratio, bool pre_or_postLayerNorm,
    std::string activation_fn, FeedForwardLayerWeightPtr ffn_wt)
    : Layer("FeedForwardLayer"),
      _layer_id(layer_id),
      _max_batch_tokens(max_batch_tokens),
      _max_seq_len(max_seq_len),
      _hidden_size(hidden_size),
      _heads(num_heads),
      _intermediate_size(intermediate_size),
      _pre_or_postLayerNorm(pre_or_postLayerNorm),
      _activation_fn(activation_fn),

      // operators
      _ffn_ln(new NormalizeLayerOp<T1, T2>(max_batch_tokens, hidden_size)),
      _ff1(new FeedForwardOp<T1, T2>(max_batch_tokens, intermediate_size,
                                     hidden_size)),
      _ffn_activation_dropout(new BiasActDropoutOp<T1, T2>(
          activation_dropout_ratio, max_batch_tokens * intermediate_size,
          activation_fn)),
      _ff2(new FeedForwardOp<T1, T2>(max_batch_tokens, hidden_size,
                                     intermediate_size)),
      _ffn_dropout(new BiasDropoutResOp<T1, T2>(
          hidden_output_dropout_ratio, max_batch_tokens * hidden_size)) {
  // parameters node
  _inter_w = new Variable(this->_name + "_inter_w", ffn_wt->_inter_w_ptr,
                          ffn_wt->_grad_inter_w_ptr);
  _inter_b = new Variable(this->_name + "_inter_b", ffn_wt->_inter_b_ptr,
                          ffn_wt->_grad_inter_b_ptr);

  _output_w =
      new Variable(this->_name + "_output_w", ffn_wt->_output_w_ptr,
                   ffn_wt->_grad_output_w_ptr);
  _output_b =
      new Variable(this->_name + "_output_b", ffn_wt->_output_b_ptr,
                   ffn_wt->_grad_output_b_ptr);

  _ffn_nw = new Variable(this->_name + "_ffn_nw", ffn_wt->_ffn_nw_ptr,
                         ffn_wt->_grad_ffn_nw_ptr);
  _ffn_nb = new Variable(this->_name + "_ffn_nb", ffn_wt->_ffn_nb_ptr,
                         ffn_wt->_grad_ffn_nb_ptr);

  this->_context_ptr->exit_layer();  // necessary
}

template <typename T1, typename T2>
Variable* FeedForwardLayer<T1, T2>::operator()(Variable* inp) {
  this->set_inputs({inp});
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
    this->set_outputs({ffn_ln_out});
    return ffn_ln_out;
  } else {
    this->set_outputs({ffn_dropout});
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
void FeedForwardLayer<T1, T2>::before_backward() {}

// template class FeedForwardLayer<float, float>;
// template class FeedForwardLayer<__half, __half>;

}  // namespace lightseq
