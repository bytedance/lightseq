#include "softmax.h"

namespace lightseq {

template <typename T1, typename T2>
Variable* SoftmaxOp<T1, T2>::operator()(Variable* inp, Variable* mask) {
  size_t max_ele_num = _max_batch_tokens * _max_seq_len * _nhead;
  Variable* result = new Variable("SoftmaxOp_out", inp);

  if (mask != nullptr)
    set_parents({inp, mask});
  else
    set_parents({inp});

  this->set_children({result});
  return result;
}

template <typename T1, typename T2>
void SoftmaxOp<T1, T2>::forward() {
  cudaStream_t stream = _context_ptr->get_stream();

  T1* inp_ptr = (T1*)parent(0)->value(true);
  T1* mask_ptr = _parents.size() > 1 ? (T1*)parent(1)->value() : nullptr;
  T1* out_ptr = (T1*)child(0)->value();

  if(!_context_ptr->is_built()){
    return ;
  }

  launch_attn_softmax_new<T1>(out_ptr, inp_ptr, mask_ptr, _batchs, _nhead,
                              _from_len, _to_len,
                              _config_mask_future | _mask_future, stream);
}

template <typename T1, typename T2>
void SoftmaxOp<T1, T2>::backward() {
  cudaStream_t stream = _context_ptr->get_stream();

  T1* soft_out = (T1*)child(0)->value();
  T2* out_grad = (T2*)child(0)->grad();
  T2* inp_grad = (T2*)parent(0)->grad();

  if(!_context_ptr->is_built()){
    return ;
  }

  launch_attn_softmax_bw_new<T2>(inp_grad, out_grad, soft_out,
                                 _batchs * _nhead * _from_len, _to_len, stream);
}

template class SoftmaxOp<float, float>;
template class SoftmaxOp<__half, __half>;

}  // namespace lightseq
