#include "transform_0213.h"

namespace lightseq {

template <typename T1, typename T2>
Variable* Transform0213OP<T1, T2>::operator()(Variable* inp) {
  Variable* res = new Variable("Transform0213_res", _max_numel * sizeof(T1),
                               _max_numel * sizeof(T2));
  set_parents({inp});
  this->set_children({res});
  return res;
}

template <typename T1, typename T2>
void Transform0213OP<T1, T2>::forward() {
  cudaStream_t _stream = _context_ptr->get_stream();

  T1* inp_ptr = (T1*)parent(0)->value();
  T1* res_ptr = (T1*)child(0)->value();

  if (!_context_ptr->is_built()) {
    return;
  }

  // [sz0, sz1, sz2, sz3] -> [sz0, sz2, sz1, sz3]
  launch_transform_0213<T2>(inp_ptr, res_ptr, _sz0, _sz1, _sz2, _sz3, _stream);
}

template <typename T1, typename T2>
void Transform0213OP<T1, T2>::backward() {
  cudaStream_t _stream = _context_ptr->get_stream();
  T2* inp_grad = (T1*)parent(0)->grad();
  T2* out_grad = (T1*)child(0)->grad();

  if (!_context_ptr->is_built()) {
    return;
  }

  // [sz0, sz1, sz2, sz3] -> [sz0, sz2, sz1, sz3]
  launch_transform_0213<T2>(out_grad, inp_grad, _sz0, _sz1, _sz2, _sz3,
                            _stream);
}

template class Transform0213OP<float, float>;
template class Transform0213OP<__half, __half>;

}  // namespace lightseq
