#include "llama_mlp_layer.h"

namespace lightseq {

template <typename T1, typename T2>
LlamaMLPLayer<T1, T2>::LlamaMLPLayer(int max_batch_tokens, int hidden_dim,
                                     int inner_dim)
    : Layer("LlamaMLPLayer"),
      _max_batch_tokens(max_batch_tokens),
      _hidden_dim(hidden_dim),
      _inner_dim(inner_dim),
      _mlp_ln(new RMSLayerNormalizeOp<T1, T2>(max_batch_tokens, hidden_dim)),
      _gate_up_linear(
          new LinearOp<T1, T2>(max_batch_tokens, 2 * inner_dim, hidden_dim)),
      _act_product(
          new ActElewiseProductOp<T1, T2>(max_batch_tokens, inner_dim)),
      _down_linear(
          new LinearOp<T1, T2>(max_batch_tokens, hidden_dim, inner_dim)),
      _add_residual(new FuseAdd2Op<T1, T2>(max_batch_tokens, hidden_dim)) {

  _norm_scale = new Variable("_norm_scale", g_dtype<T1>(), g_dtype<T2>());
  _gate_up_linear_weight =
      new Variable("_gate_up_linear_weight", g_dtype<T1>(), g_dtype<T2>());
  _down_linear_weight =
      new Variable("_down_linear_weight", g_dtype<T1>(), g_dtype<T2>());
  this->_context_ptr->exit_layer();  // necessary
}

template <typename T1, typename T2>
Variable* LlamaMLPLayer<T1, T2>::operator()(Variable* inp) {
  set_inputs({inp});
  Variable* ln_out = (*_mlp_ln)(inp, _norm_scale);
  Variable* gate_up_out = (*_gate_up_linear)(ln_out, _gate_up_linear_weight);
  Variable* act_out = (*_act_product)(gate_up_out);
  Variable* down_out = (*_down_linear)(act_out, _down_linear_weight);
  Variable* mlp_out = (*_add_residual)(down_out, inp);
  set_outputs({mlp_out});
  return mlp_out;
}

template <typename T1, typename T2>
void LlamaMLPLayer<T1, T2>::before_forward(int batch_size, int seq_len) {
  _mlp_ln->before_forward(batch_size, seq_len);
  _gate_up_linear->before_forward(batch_size * seq_len);
  _act_product->before_forward(batch_size, seq_len);
  _down_linear->before_forward(batch_size * seq_len);
  _add_residual->before_forward(batch_size, seq_len);
}

template <typename T1, typename T2>
int LlamaMLPLayer<T1, T2>::load_params(const std::vector<const T1*>& para_vec,
                                       int offset) {
  int size = 0;

  _norm_scale->set_value((char*)para_vec[offset + size]), size++;
  _norm_scale->set_shape({_hidden_dim});
  
  _gate_up_linear_weight->set_value((char*)para_vec[offset + size]), size++;
  _gate_up_linear_weight->set_shape({_hidden_dim, 2 * _inner_dim});

  _down_linear_weight->set_value((char*)para_vec[offset + size]), size++;
  _down_linear_weight->set_shape({_inner_dim, _hidden_dim});

  return size;
}

}  // namespace lightseq
