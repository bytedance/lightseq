#include "dropout.h"

namespace lightseq {

template <typename T1, typename T2>
Variable* DropoutOp<T1, T2>::operator()(Variable* inp) {
  Variable* result = new Variable("DropoutOp_out", _max_ele_num * sizeof(T1),
                                  _max_ele_num * sizeof(T2));
  this->set_parents({inp});
  this->set_children({result});
  return result;
}

template <typename T1, typename T2>
void DropoutOp<T1, T2>::forward() {
  cudaStream_t stream = _context_ptr->get_stream();

  T1* input = (T1*)parent(0)->value();
  T1* output = (T1*)child(0)->value();
  uint8_t* mask_ptr = (uint8_t*)_mask->tensor();

  launch_ls_dropout<T1>(output, input, mask_ptr, _count, RATIO(), stream,
                        false);
}

template <typename T1, typename T2>
void DropoutOp<T1, T2>::backward() {
  cudaStream_t stream = _context_ptr->get_stream();

  T2* input_grad = (T2*)parent(0)->grad();
  T2* output_grad = (T2*)child(0)->grad();
  uint8_t* mask_ptr = (uint8_t*)_mask->tensor();

  launch_ls_dropout<T2>(input_grad, output_grad, mask_ptr, _count, RATIO(),
                        stream, true);
}

template class DropoutOp<float, float>;
template class DropoutOp<__half, __half>;

}  // namespace lightseq
