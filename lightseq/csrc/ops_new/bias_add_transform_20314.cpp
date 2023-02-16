#include "bias_add_transform_20314.h"

namespace lightseq {

template <typename T1, typename T2>
Variable* BiasAddTrans20314<T1, T2>::operator()(Variable* inp, Variable* bias) {
  // size_t trans_size = _max_batch_tokens * _hidden_size;
  _res = new Variable("BiasAddTrans20314_res",
                      _trans_count * _max_batch_tokens * _hidden_size,
                      g_dtype<T1>(), g_dtype<T2>());
  set_parents({inp, bias});
  this->set_children({_res});
  return _res;
}

template <typename T1, typename T2>
void BiasAddTrans20314<T1, T2>::forward() {
  T1* inp_ptr = (T1*)parent(0)->value();
  T1* bias_ptr = (T1*)parent(1)->value();

  T1* res_ptr = (T1*)child(0)->value();

  if (!_context_ptr->is_built()) {
    return;
  }

#ifdef LIGHTSEQ_cuda
  cudaStream_t _stream = _context_ptr->get_stream();
  cuda::launch_bias_add_transform_20314<T1>(res_ptr, inp_ptr, bias_ptr, _batch,
                                            _seq_len, _trans_count, _heads,
                                            _hidden_size / _heads, _stream);
#endif
}

template <typename T1, typename T2>
void BiasAddTrans20314<T1, T2>::backward() {
  T2* inp_grad = (T2*)parent(0)->grad();
  T2* res_grad = (T2*)child(0)->grad();
  T2* qkv_bias_grad = (T2*)parent(1)->grad();

  if (!_context_ptr->is_built()) {
    return;
  }

#ifdef LIGHTSEQ_cuda
  cudaStream_t _stream = _context_ptr->get_stream();
  cuda::launch_transform4d_0213<T2>(inp_grad, res_grad, _batch, _seq_len,
                                    _hidden_size, _heads, _trans_count,
                                    _stream);
  // calculate bias
  cuda::launch_fuse_transpose_bias_kernel<T2>(
      inp_grad, qkv_bias_grad, _batch * _seq_len, 3 * _hidden_size, _stream);
#endif
}

template class BiasAddTrans20314<float, float>;
#ifdef LIGHTSEQ_cuda
template class BiasAddTrans20314<__half, __half>;
#endif
}  // namespace lightseq
