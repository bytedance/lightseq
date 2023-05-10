#include "rms_layer_norm.h"

namespace lightseq {

template <typename T1, typename T2>
RMSLayerNormalizeOp<T1, T2>::~RMSLayerNormalizeOp() {}

template <typename T1, typename T2>
std::tuple<Variable*, Variable*> RMSLayerNormalizeOp<T1, T2>::operator()(
    Variable* inp, Variable* scale) {
  size_t max_size = _max_batch_tokens * _hidden_dim;
  _result =
      new Variable("RMSLayerNormalizeOp_out", _max_batch_tokens * _hidden_dim,
                   g_dtype<T1>(), g_dtype<T2>());
  _residual =
      new Variable("RMSLayerNormalizeOp_res", _max_batch_tokens * _hidden_dim,
                   g_dtype<T1>(), g_dtype<T2>());
  set_parents({inp, scale});
  this->set_children({_result, _residual});
  return std::make_tuple(_result, _residual);
}

template <typename T1, typename T2>
void RMSLayerNormalizeOp<T1, T2>::before_forward(size_t batch_size,
                                                 size_t seq_len) {
  _batch_tokens = batch_size * seq_len;
  _result->set_shape({batch_size, seq_len, _hidden_dim});
  if (_use_residual) {
    _residual->set_shape({batch_size, seq_len, _hidden_dim});
  }
}

template <typename T1, typename T2>
void RMSLayerNormalizeOp<T1, T2>::forward() {
  T1* inp_val = (T1*)parent(0)->value();
  T1* scale_val = (T1*)parent(1)->value();
  T1* out_val = (T1*)child(0)->value();
  T1* res_val = nullptr;
  if (_use_residual) {
    res_val = (T1*)child(1)->value();
  }
  T1* rms_vars_val = (T1*)_rms_vars->tensor();

  if (!_context_ptr->is_built()) {
    return;
  }

#ifdef LIGHTSEQ_cuda
  cudaStream_t stream = _context_ptr->get_stream();
  cuda::launch_rms_layer_norm(inp_val, scale_val, out_val, res_val,
                              rms_vars_val, _batch_tokens, _hidden_dim, stream);
#endif
}

template class RMSLayerNormalizeOp<float, float>;
#ifdef LIGHTSEQ_cuda
template class RMSLayerNormalizeOp<__half, __half>;
#endif
}  // namespace lightseq
