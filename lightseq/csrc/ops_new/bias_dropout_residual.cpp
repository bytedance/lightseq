#include "bias_dropout_residual.h"

namespace lightseq {

template <typename T1, typename T2>
Variable* BiasDropoutResOp<T1, T2>::operator()(Variable* inp, Variable* bias,
                                               Variable* residual) {
  _result = new Variable("BiasDropoutResOp_out", _max_rows * _max_cols,
                         g_dtype<T1>(), g_dtype<T2>());
  set_parents({inp, bias, residual});
  this->set_children({_result});
  return _result;
}

template <typename T1, typename T2>
void BiasDropoutResOp<T1, T2>::forward() {
  T1* input = (T1*)parent(0)->value();
  T1* bias = (T1*)parent(1)->value();
  T1* residual = (T1*)parent(2)->value();
  T1* output = (T1*)child(0)->value();
  uint8_t* mask_ptr = (uint8_t*)_mask->tensor();

  if (!_context_ptr->is_built()) {
    return;
  }

#ifdef LIGHTSEQ_cuda
  cudaStream_t stream = _context_ptr->get_stream();
  cuda::launch_ls_dropout_res_bias<T1>(output, input, mask_ptr, bias, residual,
                                       _rows * _cols, _cols, RATIO(), stream);
#endif
}

template <typename T1, typename T2>
void BiasDropoutResOp<T1, T2>::backward() {
  T2* input_grad = (T2*)parent(0)->grad();
  T2* bias_grad = (T2*)parent(1)->grad();
  T2* residual_grad = (T2*)parent(2)->grad();

  T2* output_grad = (T2*)child(0)->grad();

  uint8_t* mask_ptr = (uint8_t*)_mask->tensor();

  bool is_res_cover = parent(2)->is_cover();

  if (!_context_ptr->is_built()) {
    return;
  }

#ifdef LIGHTSEQ_cuda
  cudaStream_t stream = _context_ptr->get_stream();
  cuda::launch_ls_dropout_bias_bwd<T2>(input_grad, bias_grad, output_grad,
                                       mask_ptr, _rows, _cols, RATIO(), stream);

  if (is_res_cover) {  // cover
    CHECK_GPU_ERROR(cudaMemcpyAsync((void*)residual_grad, (void*)output_grad,
                                    _cols * _rows * sizeof(T2),
                                    cudaMemcpyDefault, stream));
  } else {
    cuda::launch_fused_add2(residual_grad, output_grad, residual_grad, _rows, 1,
                            _cols, stream);
  }
#endif
}

template class BiasDropoutResOp<float, float>;
#ifdef LIGHTSEQ_cuda
template class BiasDropoutResOp<__half, __half>;
#endif
}  // namespace lightseq
