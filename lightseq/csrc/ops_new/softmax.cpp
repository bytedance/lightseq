#include "softmax.h"

namespace lightseq {

template <typename T1, typename T2>
Variable* SoftmaxOp<T1, T2>::operator()(Variable* inp) {
  Variable* result =
      new Variable(this->_name + "/out", _max_ele_num * sizeof(T1),
                   _max_ele_num * sizeof(T2));
  this->set_parents({inp});
  this->set_children({result});
  return result;
}

template <typename T1, typename T2>
void SoftmaxOp<T1, T2>::forward() {
  cudaStream_t stream = _context_ptr->get_stream();

  // need to modify launch_attn_softmax

  launch_attn_softmax<T1>(vals, attn_mask, batch_size, config_.nhead, from_len,
                         to_len, config_.mask_future | mask_future, stream);
}

template <typename T1, typename T2>
void SoftmaxOp<T1, T2>::backward() {
  cudaStream_t stream = _context_ptr->get_stream();

  T1* soft_out = (T1*)child(0)->value();
  T2* out_grad = (T2*)child(0)->grad();
  T2* inp_grad = (T2*)parent(0)->grad();

  // need to modify launch_attn_softmax_bw

  launch_attn_softmax_bw<T>(out_grad, soft_out,
                            batch_size * config_.nhead * from_len, to_len,
                            stream);
}

template class SoftmaxOp<float, float>;
template class SoftmaxOp<__half, __half>;

}  // namespace lightseq
