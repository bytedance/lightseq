#include "fuse_rotary_position_qkv.h"

namespace lightseq {

template <typename T1, typename T2>
Variable* RotaryPositionQk<T1, T2>::operator()(Variable* inp, Variable* cache_k,
                                               Variable* cache_v) {
  size_t max_size = _max_batch_size * _max_step * _head_num * _head_dim;
  _result = new Variable("RotaryPositionQk_out", max_size, g_dtype<T1>(),
                         g_dtype<T2>());
  set_parents({inp, cache_k, cache_v});
  this->set_children({_result});
  return _result;
}

template <typename T1, typename T2>
void RotaryPositionQk<T1, T2>::forward() {
  T1* inp_val = (T1*)parent(0)->value();
  T1* cache_k_val = (T1*)parent(1)->value();
  T1* cache_v_val = (T1*)parent(2)->value();

  T1* out_val = (T1*)child(0)->value();

  if (!_context_ptr->is_built()) {
    return;
  }

#ifdef LIGHTSEQ_cuda
  cudaStream_t stream = _context_ptr->get_stream();
  cuda::launch_split_rotary_position_qkv(
      inp_val, _device_sin_ptr, _device_cos_ptr, out_val, cache_k_val,
      cache_v_val, _max_step, _batch_size, _head_num, _offset_seq_len,
      _query_len, _head_dim, stream);
#endif
}

template class RotaryPositionQk<float, float>;
#ifdef LIGHTSEQ_cuda
template class RotaryPositionQk<__half, __half>;
#endif
}  // namespace lightseq
