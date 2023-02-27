#include "linear_layer.h"

namespace lightseq {

template <typename T1, typename T2>
LinearLayer<T1, T2>::LinearLayer(int max_batch_tokens, int input_size,
                                 int output_size, MATRIX_OP opA, MATRIX_OP opB,
                                 float alpha)
    : Layer("LinearLayer"),
      _max_batch_tokens(max_batch_tokens),
      _input_size(input_size),
      _output_size(output_size),
      // operators
      _linear(new LinearOp<T1, T2>(max_batch_tokens, output_size, input_size,
                                   opA, opB, alpha)) {
  // parameters node
  _linear_w = new Variable("_linear_w", g_dtype<T1>(), g_dtype<T2>());

  this->_context_ptr->exit_layer();  // necessary
}

template <typename T1, typename T2>
Variable* LinearLayer<T1, T2>::operator()(Variable* inp) {
  set_inputs({inp});
  Variable* linear_out = (*_linear)(inp, _linear_w);

  set_outputs({linear_out});
  return linear_out;
}

template <typename T1, typename T2>
void LinearLayer<T1, T2>::before_forward(int batch_size, int seq_len) {
  int batch_tokens = batch_size * seq_len;

  _linear->before_forward(batch_tokens);
}

template <typename T1, typename T2>
void LinearLayer<T1, T2>::before_backward() {}

template <typename T1, typename T2>
size_t LinearLayer<T1, T2>::load_para_and_grad(const T1* para_ptr,
                                               T2* grad_ptr) {  // for training
  size_t offset = 0;

  _linear_w->set_value((char*)(para_ptr + offset));
  _linear_w->set_grad((char*)(grad_ptr + offset));
  _linear_w->set_shape({_output_size, _input_size});
  offset += _input_size * _output_size;

  return offset;
}

template <typename T1, typename T2>
int LinearLayer<T1, T2>::load_params(const std::vector<const T1*>& para_vec,
                                     int offset) {  // for inference
  int size = 0;
  _linear_w->set_value((char*)para_vec[offset + size]), size++;

  return size;
}

}  // namespace lightseq
