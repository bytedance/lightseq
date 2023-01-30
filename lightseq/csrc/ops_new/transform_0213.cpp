#include "transform_0213.h"

namespace lightseq {

template <typename T1, typename T2>
Variable* Transform0213<T1, T2>::operator()(Variable* inp) {
  size_t trans_size = _max_batch_tokens * _hidden_size;
  Variable* res = new Variable("Transform0213_res", trans_size * sizeof(T1),
                               trans_size * sizeof(T2));
  set_parents({inp});
  this->set_children({res});
  return res;
}

template <typename T1, typename T2>
void Transform0213<T1, T2>::forward() {
  T1* inp_ptr = (T1*)parent(0)->value();
  T1* res_ptr = (T1*)child(0)->value();

  if (!_context_ptr->is_built()) {
    return;
  }

#ifdef LIGHTSEQ_cuda
  cudaStream_t _stream = _context_ptr->get_stream();
  //   [b, nh, s, ad] -> [b, s, nh, ad]
  launch_transform4d_0213<T1>(res_ptr, inp_ptr, _batch, _seq_len, _hidden_size,
                              _heads, 1, _stream);
#endif
}

template <typename T1, typename T2>
void Transform0213<T1, T2>::backward() {
  T2* inp_grad = (T1*)parent(0)->grad();
  T2* out_grad = (T1*)child(0)->grad();

  if (!_context_ptr->is_built()) {
    return;
  }

#ifdef LIGHTSEQ_cuda
  cudaStream_t _stream = _context_ptr->get_stream();
  launch_transform_0213<T2>(inp_grad, out_grad, _batch, _seq_len, _hidden_size,
                            _heads, _stream);
#endif
}

template class Transform0213<float, float>;
#ifdef LIGHTSEQ_cuda
template class Transform0213<__half, __half>;
#endif
}  // namespace lightseq
