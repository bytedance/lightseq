#include "act_elewise_product.h"

namespace lightseq {

template <typename T1, typename T2>
Variable* ActElewiseProductOp<T1, T2>::operator()(Variable* inp) {
  size_t max_size = _max_batch_tokens * _inner_size;
  _result = new Variable("ActElewiseProductOp_out", max_size, g_dtype<T1>(),
                         g_dtype<T2>());
  set_parents({inp});
  this->set_children({_result});
  return _result;
}

template <typename T1, typename T2>
void ActElewiseProductOp<T1, T2>::forward() {
  T1* inp_val = (T1*)parent(0)->value();
  T1* out_val = (T1*)child(0)->value();

  if (!_context_ptr->is_built()) {
    return;
  }

#ifdef LIGHTSEQ_cuda
  cudaStream_t stream = _context_ptr->get_stream();
  cuda::launch_silu_elewise_product(inp_val, out_val, _batch_size, _seq_len,
                                    _inner_size, stream);
#endif
}

template class ActElewiseProductOp<float, float>;
#ifdef LIGHTSEQ_cuda
template class ActElewiseProductOp<__half, __half>;
#endif
}  // namespace lightseq
