#include "feed_forward.h"

namespace lightseq {

template <typename T1, typename T2>
Variable* FeedForwardOp<T1, T2>::operator()(Variable* inp, Variable* weight) {
  size_t max_size = _max_batch_tokens * _output_size;
  Variable* result = new Variable(this->_name + "/out", max_size * sizeof(T1),
                                  max_size * sizeof(T2));
  this->set_parents({inp, weight});
  this->set_children({result});
  return result;
}

template <typename T1, typename T2>
void FeedForwardOp<T1, T2>::forward() {
  float alpha = float(1.);
  float beta = float(0.);

  T1* weights = (T1*)parent(1)->value();
  T1* input_ptr = (T1*)parent(0)->value();
  T1* out_ptr = (T1*)child(0)->value();
  cublasHandle_t _cublasHandle = _context_ptr->get_cublashandle();

  cublas_gemm_ex(_cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, _output_size,
                 _batch_tokens, _input_size, &alpha, &beta, weights, input_ptr,
                 out_ptr, cublasGemmAlgo_t(_gemm_algos[0]));
}

template <typename T1, typename T2>
void FeedForwardOp<T1, T2>::backward() {
  float alpha = (float)1.0, w_beta = (float)0.0, inp_beta = (float)0.0;

  T2* out_grad = (T2*)child(0)->grad();
  T1* weights = (T1*)parent(1)->value();
  T1* input_ptr = (T1*)parent(0)->value();

  T2* weights_grad = (T2*)parent(1)->grad();
  T2* inp_grad = (T2*)parent(0)->grad();

  if (!parent(0)->is_cover()) {
    inp_beta = (float)1.0;
  }

  cublasHandle_t _cublasHandle = _context_ptr->get_cublashandle();

  cublas_gemm_ex(_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, _input_size,
                 _output_size, _batch_tokens, &alpha, &w_beta, input_ptr,
                 out_grad, weights_grad, cublasGemmAlgo_t(_gemm_algos[1]));

  cublas_gemm_ex(_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, _input_size,
                 _batch_tokens, _output_size, &alpha, &inp_beta, weights,
                 out_grad, inp_grad, cublasGemmAlgo_t(_gemm_algos[2]));
}

template class FeedForwardOp<float, float>;
template class FeedForwardOp<__half, __half>;

}  // namespace lightseq
