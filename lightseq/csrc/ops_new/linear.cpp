#include "linear.h"

namespace lightseq {

template <typename T1, typename T2>
Variable* LinearOp<T1, T2>::operator()(Variable* inp, Variable* weight) {
  // size_t max_size = _max_batch_tokens * _output_size;
  _result = new Variable("LinearOp_out", _max_batch_tokens * _output_size,
                         g_dtype<T1>(), g_dtype<T2>());
  set_parents({inp, weight});
  this->set_children({_result});
  return _result;
}

template <typename T1, typename T2>
void LinearOp<T1, T2>::forward() {
  T1* input_ptr = (T1*)parent(0)->value();
  T1* weights = (T1*)parent(1)->value();
  T1* out_ptr = (T1*)child(0)->value();

  if (!_context_ptr->is_built()) {
    return;
  }

#ifdef LIGHTSEQ_cuda
  cublasHandle_t _cublasHandle = _context_ptr->get_cublashandle();
  cuda::cublas_gemm_ex(_cublasHandle, op_from_custom(_opA),
                       op_from_custom(_opB), _output_size, _batch_tokens,
                       _input_size, &_alpha, &_beta, weights, input_ptr,
                       out_ptr, cublasGemmAlgo_t(_gemm_algos[0]));
#elif defined LIGHTSEQ_x86
  x86::matrix_gemm(weights, input_ptr, out_ptr, _output_size, _batch_tokens,
                   _input_size);
#endif
}

template <typename T1, typename T2>
void LinearOp<T1, T2>::backward() {
  float bw_alpha = 1. / _alpha;
  float w_beta = (float)0.0, inp_beta = (float)0.0;

  T2* out_grad = (T2*)child(0)->grad();
  T1* input_ptr = (T1*)parent(0)->value();
  T1* weights = (T1*)parent(1)->value();

  T2* inp_grad = (T2*)parent(0)->grad();
  T2* weights_grad = (T2*)parent(1)->grad();

  if (!parent(0)->is_cover()) {
    inp_beta = (float)1.0;
  }

  if (!_context_ptr->is_built()) {
    return;
  }

#ifdef LIGHTSEQ_cuda
  cublasHandle_t _cublasHandle = _context_ptr->get_cublashandle();
  // Q: how to adpat _opA & _opB
  // calculate weights_grad
  cuda::cublas_gemm_ex(_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, _input_size,
                       _output_size, _batch_tokens, &bw_alpha, &w_beta,
                       input_ptr, out_grad, weights_grad,
                       cublasGemmAlgo_t(_gemm_algos[1]));

  // calculate inp_grad
  cuda::cublas_gemm_ex(_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, _input_size,
                       _batch_tokens, _output_size, &bw_alpha, &inp_beta,
                       weights, out_grad, inp_grad,
                       cublasGemmAlgo_t(_gemm_algos[2]));
#endif
}

template class LinearOp<float, float>;
#ifdef LIGHTSEQ_cuda
template class LinearOp<__half, __half>;
#endif
}  // namespace lightseq
