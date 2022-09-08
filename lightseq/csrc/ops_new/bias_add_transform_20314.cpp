#include "bias_add_transform_20314.h"

namespace lightseq {

template <typename T1, typename T2>
std::tuple<Variable*, Variable*, Variable*>
BiasAddTrans20314<T1, T2>::operator()(Variable* inp, Variable* bias) {
  size_t trans_size = _max_batch_tokens * _hidden_size;
  Variable* res0 =
      new Variable("BiasAddTrans20314_res_0", trans_size * sizeof(T1),
                   trans_size * sizeof(T2));
  Variable* res1 =
      new Variable("BiasAddTrans20314_res_1", trans_size * sizeof(T1),
                   trans_size * sizeof(T2));
  Variable* res2 =
      new Variable("BiasAddTrans20314_res_2", trans_size * sizeof(T1),
                   trans_size * sizeof(T2));
  this->set_parents({inp, bias});
  this->set_children({res0, res1, res2});
  return std::make_tuple(res0, res1, res2);
}

template <typename T1, typename T2>
void BiasAddTrans20314<T1, T2>::forward() {
  cudaStream_t _stream = _context_ptr->get_stream();

  T1* inp_ptr = (T1*)parent(0)->value();
  T1* bias_ptr = (T1*)parent(1)->value();

  T1* q_ptr = (T1*)child(0)->value();
  T1* k_ptr = (_trans_count <= 1) ? nullptr : (T1*)child(1)->value();
  T1* v_ptr = (_trans_count <= 2) ? nullptr : (T1*)child(2)->value();

  launch_bias_add_transform_20314_new<T1>(
      q_ptr, k_ptr, v_ptr, inp_ptr, bias_ptr, _batch, _seq_len, _trans_count,
      _heads, _hidden_size / _heads, _stream);
}

template <typename T1, typename T2>
void BiasAddTrans20314<T1, T2>::backward() {
  cudaStream_t _stream = _context_ptr->get_stream();
  T2* inp_grad = (T2*)parent(0)->grad();
  T2* q_grad = (T2*)child(0)->grad();
  T2* k_grad = (_trans_count <= 1) ? nullptr : (T2*)child(1)->grad();
  T2* v_grad = (_trans_count <= 2) ? nullptr : (T2*)child(2)->grad();

  launch_transform_20314_bwd_new<T2>(inp_grad, q_grad, k_grad, v_grad, _batch,
                                     _seq_len, _hidden_size, _heads,
                                     _trans_count, _stream);

  // calculate bias
  T2* qkv_bias_grad = (T2*)parent(1)->grad();

  launch_fuse_transpose_bias_kernel<T2>(
      inp_grad, qkv_bias_grad, _batch * _seq_len, 3 * _hidden_size, _stream);
}

template class BiasAddTrans20314<float, float>;
template class BiasAddTrans20314<__half, __half>;

}  // namespace lightseq
