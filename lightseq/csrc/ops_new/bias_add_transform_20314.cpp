#include "bias_add_transform_20314.h"

namespace lightseq {

template <typename T1, typename T2>
std::tuple<Variable*, Variable*, Variable*>
BiasAddTrans20314<T1, T2>::operator()(Variable* inp, Variable* bias) {
  size_t trans_size = _max_batch_tokens * _hidden_size;
  Variable* res_q = new Variable(
      this->_name + "/res_q", trans_size * sizeof(T1), trans_size * sizeof(T2));
  Variable* res_k = new Variable(
      this->_name + "/res_k", trans_size * sizeof(T1), trans_size * sizeof(T2));
  Variable* res_v = new Variable(
      this->_name + "/res_v", trans_size * sizeof(T1), trans_size * sizeof(T2));
  this->set_parents({inp, bias});
  this->set_children({res_q, res_k, res_v});
  return std::make_tuple(res_q, res_k, res_v);
}

template <typename T1, typename T2>
void BiasAddTrans20314<T1, T2>::forward() {
  cudaStream_t _stream = _context_ptr->get_stream();

  T1* inp_ptr = (T1*)parent(0)->value();
  T1* bias_ptr = (T1*)parent(1)->value();

  T1* q_ptr = (T1*)child(0)->value();
  T1* k_ptr = (T1*)child(1)->value();
  T1* v_ptr = (T1*)child(2)->value();

  launch_bias_add_transform_20314_new<T1>(q_ptr, k_ptr, v_ptr, inp_ptr,
                                          bias_ptr, _batch, _seq_len, 3, _heads,
                                          _hidden_size / _heads, _stream);

#ifdef DEBUG
  if (_context_ptr->built()) {
    cudaStreamSynchronize(_stream);
    print_vec(q_ptr, "after_transform q", 10);
    print_vec(k_ptr, "after_transform k", 10);
    print_vec(v_ptr, "after_transform v", 10);
    printf("\n");
  }
#endif
}

template <typename T1, typename T2>
void BiasAddTrans20314<T1, T2>::backward() {
  cudaStream_t _stream = _context_ptr->get_stream();
  T2* inp_grad = (T1*)parent(0)->grad();
  T2* q_grad = (T1*)child(0)->grad();
  T2* k_grad = (T1*)child(1)->grad();
  T2* v_grad = (T1*)child(2)->grad();

  launch_transform_20314_bwd_new<T2>(inp_grad, q_grad, k_grad, v_grad, _batch,
                                     _seq_len, _hidden_size, _heads, _stream);

  // calculate bias
  T2* qkv_bias_grad = (T2*)parent(1)->grad();

  launch_fuse_transpose_bias_kernel<T2>(
      inp_grad, qkv_bias_grad, _batch * _seq_len, 3 * _hidden_size, _stream);
}

template class BiasAddTrans20314<float, float>;
template class BiasAddTrans20314<__half, __half>;

}  // namespace lightseq
