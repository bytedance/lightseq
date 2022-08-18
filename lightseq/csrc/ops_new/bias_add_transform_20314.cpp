#include "bias_add_transform_20314.h"

namespace lightseq {

template <typename T1, typename T2>
Variable* BiasAddTrans20314<T1, T2>::operator()(Variable* inp, Variable* bias) {
    size_t trans_size = _max_batch * _max_seq * _hidden_size;
    Variable* res_q =
      new Variable(this->_name + "/res_q", trans_size * sizeof(T1), trans_size * sizeof(T2));
    Variable* res_k = 
      new Variable(this->_name + "/res_k", trans_size * sizeof(T1), trans_size * sizeof(T2));
    Variable* res_v = 
      new Variable(this->_name + "/res_v", trans_size * sizeof(T1), trans_size * sizeof(T2));
    this->set_parents({inp, bias});
    this->set_children({res_q, res_k, res_v});
    return result;
}

template <typename T1, typename T2>
void BiasAddTrans20314<T1, T2>::forward() {
  cudaStream_t _stream = _context_ptr->get_stream();

  T1* inp_ptr = (T1*)parent(0)->value();
  T1* bias_ptr = (T1*)parent(0)->value();

  T1* q_ptr = (T1*)child(0)->value();
  T1* k_ptr = (T1*)child(1)->value();
  T1* v_ptr = (T1*)child(2)->value();

// TODO: add launch_bias_add_transform_20314_new 
//   launch_bias_add_transform_20314<T>(q_ptr, buffer, _attn_qkvb_ptr,
//                                      _batch_size, _seq_len, 3, _heads,
//                                      _hidden_size / _heads, _stream);
}

template <typename T1, typename T2>
void BiasAddTrans20314<T1, T2>::backward() {

    T2* inp_grad = (T1*)parent(0)->grad();
    T2* q_grad = (T1*)child(0)->grad();
    T2* k_grad = (T1*)child(1)->grad();
    T2* v_grad = (T1*)child(2)->grad();

    // TODO: add launch_transform4d_0213_new
    // launch_transform4d_0213<T>(grad_qkv_4d_ptr, grad_qkv_5d_ptr, _batch_size,
    //                          _seq_len, _hidden_size, _heads, 3, _stream);

    // calculate bias
    T2* qkv_bias_grad = (T2*)parent(1)->grad();
}

} // namespace lightseq 