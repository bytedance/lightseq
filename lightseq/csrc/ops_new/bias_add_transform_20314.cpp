#include "bias_add_transform_20314.h"

namespace lightseq {

template <typename T1, typename T2>
Variable* BiasAddTrans20314<T1, T2>::operator()(Variable* inp, Variable* bias) {
  size_t trans_size = _max_batch_tokens * _hidden_size;
  Variable* res = new Variable("BiasAddTrans20314_res",
                               _trans_count * trans_size * sizeof(T1),
                               _trans_count * trans_size * sizeof(T2));
  set_parents({inp, bias});
  this->set_children({res});
  return res;
}

template <typename T1, typename T2>
void BiasAddTrans20314<T1, T2>::forward() {
  cudaStream_t _stream = _context_ptr->get_stream();

  T1* inp_ptr = (T1*)parent(0)->value();
  T1* bias_ptr = (T1*)parent(1)->value();

  T1* res_ptr = (T1*)child(0)->value();

  if(!_context_ptr->is_built()){
    return ;
  }

  launch_bias_add_transform_20314<T1>(res_ptr, inp_ptr, bias_ptr, _batch,
                                      _seq_len, _trans_count, _heads,
                                      _hidden_size / _heads, _stream);
}

template <typename T1, typename T2>
void BiasAddTrans20314<T1, T2>::backward() {
  cudaStream_t _stream = _context_ptr->get_stream();
  T2* inp_grad = (T2*)parent(0)->grad();
  T2* res_grad = (T2*)child(0)->grad();
  T2* qkv_bias_grad = (T2*)parent(1)->grad();

  if(!_context_ptr->is_built()){
    return ;
  }

  launch_transform4d_0213<T2>(inp_grad, res_grad, _batch, _seq_len,
                              _hidden_size, _heads, _trans_count, _stream);

  // calculate bias
  launch_fuse_transpose_bias_kernel<T2>(
      inp_grad, qkv_bias_grad, _batch * _seq_len, 3 * _hidden_size, _stream);
}

template class BiasAddTrans20314<float, float>;
template class BiasAddTrans20314<__half, __half>;

}  // namespace lightseq
