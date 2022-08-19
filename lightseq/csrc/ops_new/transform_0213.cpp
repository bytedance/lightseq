#include "transform_0213.h"

namespace lightseq {

template <typename T1, typename T2>
Variable* Transform0213<T1, T2>::operator()(Variable* inp) {
  size_t trans_size = _max_batch * _max_seq * _hidden_size;
  Variable* res = new Variable(this->_name + "/res", trans_size * sizeof(T1),
                               trans_size * sizeof(T2));
  this->set_parents({inp});
  this->set_children({res});
  return res;
}

template <typename T1, typename T2>
void Transform0213<T1, T2>::forward() {
  cudaStream_t _stream = _context_ptr->get_stream();

  T1* inp_ptr = (T1*)parent(0)->value();
  T1* res_ptr = (T1*)child(0)->value();

  //   [b, nh, s, ad] -> [b, s, nh, ad]
  launch_transform4d_0213<T1>(res_ptr, inp_ptr, _batch, _seq_len, _hidden_size,
                              _heads, 1, _stream);
}

template <typename T1, typename T2>
void Transform0213<T1, T2>::backward() {
  cudaStream_t _stream = _context_ptr->get_stream();
  T2* inp_grad = (T1*)parent(0)->grad();
  T2* out_grad = (T1*)child(0)->grad();

  launch_transform_0213<T2>(inp_grad, out_grad, _batch, _seq_len, _hidden_size,
                            _heads, _stream);
}

}  // namespace lightseq
