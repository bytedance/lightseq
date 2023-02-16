#include "dropout.h"

namespace lightseq {

template <typename T1, typename T2>
Variable* DropoutOp<T1, T2>::operator()(Variable* inp) {
  _result =
      new Variable("DropoutOp_out", _max_ele_num, g_dtype<T1>(), g_dtype<T2>());
  set_parents({inp});
  this->set_children({_result});
  return _result;
}

template <typename T1, typename T2>
void DropoutOp<T1, T2>::forward() {
  T1* input = (T1*)parent(0)->value();
  T1* output = (T1*)child(0)->value();
  uint8_t* mask_ptr = (uint8_t*)_mask->tensor();

  if (!_context_ptr->is_built()) {
    return;
  }

  if (_is_skip) {
    return;
  }

#ifdef LIGHTSEQ_cuda
  cudaStream_t stream = _context_ptr->get_stream();
  cuda::launch_ls_dropout<T1>(output, input, mask_ptr, _count, RATIO(), stream,
                              false);
#elif defined LIGHTSEQ_x86
  //.....
#endif
}

template <typename T1, typename T2>
void DropoutOp<T1, T2>::backward() {
  T2* input_grad = (T2*)parent(0)->grad();
  T2* output_grad = (T2*)child(0)->grad();
  uint8_t* mask_ptr = (uint8_t*)_mask->tensor();

  if (!_context_ptr->is_built()) {
    return;
  }

  if (_is_skip) {
    return;
  }

#ifdef LIGHTSEQ_cuda
  cudaStream_t stream = _context_ptr->get_stream();
  cuda::launch_ls_dropout<T2>(input_grad, output_grad, mask_ptr, _count,
                              RATIO(), stream, true);
#endif
}

template class DropoutOp<float, float>;
#ifdef LIGHTSEQ_cuda
template class DropoutOp<__half, __half>;
#endif
}  // namespace lightseq
