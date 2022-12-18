#include "includes/rms_layer_normalize.h"

namespace lightseq {

template <typename T1, typename T2>
RMSLayerNormalizeOp<T1, T2>::~RMSLayerNormalizeOp() {}

template <typename T1, typename T2>
Variable* RMSLayerNormalizeOp<T1, T2>::operator()(Variable* inp, Variable* gamma,
                                               Variable* betta) {
  size_t max_size = _max_batch_tokens * _hidden_dim;
  Variable* result = new Variable("RMSLayerNormalizeOp_out", max_size * sizeof(T1),
                                  max_size * sizeof(T2));
  set_parents({inp, gamma, betta});
  this->set_children({result});
  return result;
}

template <typename T1, typename T2>
void RMSLayerNormalizeOp<T1, T2>::before_forward(size_t batch_tokens) {
  _batch_tokens = batch_tokens;
}

template <typename T1, typename T2>
void RMSLayerNormalizeOp<T1, T2>::forward() {
  T1* inp_val = (T1*)parent(0)->value();
  T1* gamma_val = (T1*)parent(1)->value();
  T1* betta_val = (T1*)parent(2)->value();
  T1* ln_res_val = (T1*)child(0)->value();
  cudaStream_t stream = _context_ptr->get_stream();

  if (!_context_ptr->is_built()) {
    return;
  }

  cuda::t5_ker_norm_layer_launcher(_batch_tokens, _hidden_dim,
                                stream, inp_val, ln_res_val,
                                gamma_val, betta_val,
                                1024);
}

template <typename T1, typename T2>
void RMSLayerNormalizeOp<T1, T2>::before_backward(size_t batch_tokens) {
  _batch_tokens = batch_tokens;
}

template <typename T1, typename T2>
void RMSLayerNormalizeOp<T1, T2>::backward() {
  T2* gamma_grad = (T2*)parent(1)->grad();
  T2* betta_grad = (T2*)parent(2)->grad();
  T2* inp_grad = (T2*)parent(0)->grad();
  T2* out_grad = (T2*)child(0)->grad();
  T2* residual_grad = nullptr;

  T1* out_val = (T1*)child(0)->value();
  T1* gamma_val = (T1*)parent(1)->value();
  T1* betta_val = (T1*)parent(2)->value();
  T1* vars_val = (T1*)vars_->tensor();

  cudaStream_t streams[2] = {_context_ptr->get_stream(),
                             _context_ptr->get_stream()};

  T1* means_val = _use_mean ? (T1*)means_->tensor() : nullptr;

  bool is_res_cover = parent(0)->is_cover();
  if (!is_res_cover) {
    residual_grad = inp_grad;
  }

  if (!_context_ptr->is_built()) {
    return;
  }

  launch_ln_bw(gamma_grad, betta_grad, inp_grad, out_grad, residual_grad,
               out_val, gamma_val, betta_val, vars_val, means_val,
               _batch_tokens, _hidden_dim, streams);
}

template class RMSLayerNormalizeOp<__half, __half>;
template class RMSLayerNormalizeOp<float, float>;

}  // namespace lightseq
