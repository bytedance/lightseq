#include "transform_0213.h"

namespace lightseq {

template <typename T1, typename T2>
Variable* Transform0213OP<T1, T2>::operator()(Variable* inp) {
  Variable* res =
      new Variable("Transform0213_res", _max_numel,
                   g_dtype<T1>(), g_dtype<T2>());
  set_parents({inp});
  this->set_children({res});
  return res;
}

template <typename T1, typename T2>
void Transform0213OP<T1, T2>::forward() {
  T1* inp_ptr = (T1*)parent(0)->value();
  T1* res_ptr = (T1*)child(0)->value();

  if (!_context_ptr->is_built()) {
    return;
  }

  //   [b, nh, s, ad] -> [b, s, nh, ad]
  launch_transform4d_0213<T1>(res_ptr, inp_ptr, _batch, _seq_len, _hidden_size,
                              _heads, 1, _stream);
}

template <typename T1, typename T2>
void Transform0213OP<T1, T2>::backward() {
  T2* inp_grad = (T1*)parent(0)->grad();
  T2* out_grad = (T1*)child(0)->grad();

  if (!_context_ptr->is_built()) {
    return;
  }

#ifdef LIGHTSEQ_cuda
  cudaStream_t _stream = _context_ptr->get_stream();
  cuda::launch_transform_0213<T2>(out_grad, inp_grad, _sz0, _sz1, _sz2, _sz3, _stream);
#endif
}

template class Transform0213OP<float, float>;
#ifdef LIGHTSEQ_cuda
template class Transform0213OP<__half, __half>;
#endif
}  // namespace lightseq
