#include "includes/layer_normalize.h"

namespace lightseq {

template <typename T1, typename T2>
LayerNormalizeOp<T1, T2>::~LayerNormalizeOp() {}

template <typename T1, typename T2>
Variable* LayerNormalizeOp<T1, T2>::operator()(Variable* inp, Variable* gamma,
                                               Variable* betta) {
  size_t max_size = _max_batch_tokens * _hidden_dim;
  Variable* result = new Variable("LayerNormalizeOp_out", max_size * sizeof(T1),
                                  max_size * sizeof(T2));
  set_parents({inp, gamma, betta});
  this->set_children({result});
  return result;
}

template <typename T1, typename T2>
void LayerNormalizeOp<T1, T2>::before_forward(size_t batch_tokens) {
  _batch_tokens = batch_tokens;
}

template <typename T1, typename T2>
void LayerNormalizeOp<T1, T2>::forward() {
  T1* ln_res_val = (T1*)child(0)->value();
  T1* inp_val = (T1*)parent(0)->value();
  T1* gamma_val = (T1*)parent(1)->value();
  T1* betta_val = (T1*)parent(2)->value();

  cudaStream_t stream = _context_ptr->get_stream();

  T1* vars_val = (T1*)vars_->tensor();
  T1* means_val = _use_mean ? (T1*)means_->tensor() : nullptr;

  launch_layer_norm(ln_res_val, vars_val, means_val, inp_val, gamma_val,
                    betta_val, _batch_tokens, _hidden_dim, stream);
}

template <typename T1, typename T2>
void LayerNormalizeOp<T1, T2>::before_backward(size_t batch_tokens) {
  _batch_tokens = batch_tokens;
}

template <typename T1, typename T2>
void LayerNormalizeOp<T1, T2>::backward() {
  T2* gamma_grad = (T2*)parent(1)->grad();
  T2* betta_grad = (T2*)parent(2)->grad();
  T2* inp_grad = (T2*)parent(0)->grad();
  T2* out_grad = (T2*)child(0)->grad();
  T2* residual_grad = nullptr;

  T1* out_val = (T1*)child(0)->value();
  T1* gamma_val = (T1*)parent(1)->value();
  T1* betta_val = (T1*)parent(2)->value();

  cudaStream_t streams[2] = {_context_ptr->get_stream(),
                             _context_ptr->get_stream()};

  T1* vars_val = (T1*)vars_->tensor();
  T1* means_val = _use_mean ? (T1*)means_->tensor() : nullptr;

  bool is_res_cover = parent(0)->is_cover();
  if (!is_res_cover) {
    residual_grad = inp_grad;
  }

  launch_ln_bw(gamma_grad, betta_grad, inp_grad, out_grad, residual_grad,
               out_val, gamma_val, betta_val, vars_val, means_val,
               _batch_tokens, _hidden_dim, streams);
}

template class LayerNormalizeOp<__half, __half>;
template class LayerNormalizeOp<float, float>;

}  // namespace lightseq
