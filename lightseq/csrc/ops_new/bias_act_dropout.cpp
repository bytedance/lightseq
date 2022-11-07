#include "bias_act_dropout.h"

namespace lightseq {

template <typename T1, typename T2>
Variable* BiasActDropoutOp<T1, T2>::operator()(Variable* inp, Variable* bias) {
  Variable* result =
      new Variable("BiasActDropoutOp_output", _max_ele_num * sizeof(T1),
                   _max_ele_num * sizeof(T2));
  set_parents({inp, bias});
  this->set_children({result});
  return result;
}

template <typename T1, typename T2>
void BiasActDropoutOp<T1, T2>::forward() {
  cudaStream_t stream = _context_ptr->get_stream();

  T1* input = (T1*)parent(0)->value();
  T1* bias = (T1*)parent(1)->value();
  T1* output = (T1*)child(0)->value();

  uint8_t* mask_ptr = (uint8_t*)_mask->tensor();

  if (!_context_ptr->is_built()) {
    return;
  }

  if (_activation_fn == "relu") {
    launch_ls_dropout_act_bias<ActivationType::kRelu, T1>(
        output, input, mask_ptr, bias, _rows * _cols, _cols, RATIO(), stream);
  } else if (_activation_fn == "gelu") {
    launch_ls_dropout_act_bias<ActivationType::kGelu, T1>(
        output, input, mask_ptr, bias, _rows * _cols, _cols, RATIO(), stream);
  } else {
    throw std::runtime_error("not supported activation: " + _activation_fn);
  }
}

template <typename T1, typename T2>
void BiasActDropoutOp<T1, T2>::backward() {
  cudaStream_t stream = _context_ptr->get_stream();

  T1* input = (T1*)parent(0)->value();
  T1* bias = (T1*)parent(1)->value();

  T2* grad_inp = (T2*)parent(0)->grad();
  T2* grad_bias = (T2*)parent(1)->grad();
  T2* grad_out = (T2*)child(0)->grad();

  uint8_t* mask_ptr = (uint8_t*)_mask->tensor();

  if (!_context_ptr->is_built()) {
    return;
  }

  if (_activation_fn == "relu") {
    launch_ls_dropout_act_bias_bwd<ActivationType::kRelu, T1>(
        grad_inp, grad_bias, input, bias, grad_out, mask_ptr, _rows, _cols,
        RATIO(), stream);
  } else if (_activation_fn == "gelu") {
    launch_ls_dropout_act_bias_bwd<ActivationType::kGelu, T1>(
        grad_inp, grad_bias, input, bias, grad_out, mask_ptr, _rows, _cols,
        RATIO(), stream);
  } else {
    throw std::runtime_error("not supported activation: " + _activation_fn);
  }
}

template class BiasActDropoutOp<float, float>;
template class BiasActDropoutOp<__half, __half>;

}  // namespace lightseq
