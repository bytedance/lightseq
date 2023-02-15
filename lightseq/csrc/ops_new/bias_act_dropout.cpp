#include "bias_act_dropout.h"

namespace lightseq {

template <typename T1, typename T2>
Variable* BiasActDropoutOp<T1, T2>::operator()(Variable* inp, Variable* bias) {
  _result = new Variable("BiasActDropoutOp_output", {_mx_rows, _mx_cols},
                         g_dtype<T1>(), g_dtype<T2>());
  set_parents({inp, bias});
  this->set_children({_result});
  return _result;
}

template <typename T1, typename T2>
void BiasActDropoutOp<T1, T2>::forward() {
  T1* input = parent(0)->value<T1>();
  T1* bias = parent(1)->value<T1>();
  T1* output = child(0)->value<T1>();

  uint8_t* mask_ptr = _mask->tensor<uint8_t>();

  if (!_context_ptr->is_built()) {
    return;
  }

#ifdef LIGHTSEQ_cuda
  cudaStream_t stream = _context_ptr->get_stream();
  if (_activation_fn == "relu") {
    cuda::launch_ls_dropout_act_bias<ActivationType::kRelu, T1>(
        output, input, mask_ptr, bias, _rows * _cols, _cols, RATIO(), stream);
  } else if (_activation_fn == "gelu") {
    cuda::launch_ls_dropout_act_bias<ActivationType::kGelu, T1>(
        output, input, mask_ptr, bias, _rows * _cols, _cols, RATIO(), stream);
  } else {
    throw std::runtime_error("not supported activation: " + _activation_fn);
  }
#endif
}

template <typename T1, typename T2>
void BiasActDropoutOp<T1, T2>::backward() {
  T1* input = parent(0)->value<T1>();
  T1* bias = parent(1)->value<T1>();

  T2* grad_inp = parent(0)->grad<T2>();
  T2* grad_bias = parent(1)->grad<T2>();
  T2* grad_out = child(0)->grad<T2>();

  uint8_t* mask_ptr = _mask->tensor<uint8_t>();

  if (!_context_ptr->is_built()) {
    return;
  }

#ifdef LIGHTSEQ_cuda
  cudaStream_t stream = _context_ptr->get_stream();
  if (_activation_fn == "relu") {
    cuda::launch_ls_dropout_act_bias_bwd<ActivationType::kRelu, T1>(
        grad_inp, grad_bias, input, bias, grad_out, mask_ptr, _rows, _cols,
        RATIO(), stream);
  } else if (_activation_fn == "gelu") {
    cuda::launch_ls_dropout_act_bias_bwd<ActivationType::kGelu, T1>(
        grad_inp, grad_bias, input, bias, grad_out, mask_ptr, _rows, _cols,
        RATIO(), stream);
  } else {
    throw std::runtime_error("not supported activation: " + _activation_fn);
  }
#endif
}

template class BiasActDropoutOp<float, float>;
#ifdef LIGHTSEQ_cuda
template class BiasActDropoutOp<__half, __half>;
#endif
}  // namespace lightseq
