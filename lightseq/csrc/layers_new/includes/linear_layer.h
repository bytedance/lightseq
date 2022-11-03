#pragma once

#include "linear.h"
#include "layer.h"

namespace lightseq {

template <class T1, class T2>
class LinearLayer : public Layer {
 private:
  // operators
  LinearOp<T1, T2>* _linear = nullptr;

  // parameters
  Variable* _linear_w;

  // shape related
  int _max_batch_tokens;
  int _input_size;
  int _output_size;

 public:
  LinearLayer(int max_batch_tokens, int input_size, int output_size, cublasOperation_t opA = CUBLAS_OP_T, cublasOperation_t opB = CUBLAS_OP_N);

  virtual ~LinearLayer() {}

  Variable* operator()(Variable* inp);

  void before_forward(int batch_size, int seq_len);

  void before_backward();

  int load_para_and_grad(const T1* para_ptr, T2* grad_ptr);

  int load_params(const std::vector<const T1*>& para_vec, int offset);
};

template class LinearLayer<__half, __half>;
template class LinearLayer<float, float>;

template <class T1, class T2>
using LinearLayerPtr = std::shared_ptr<LinearLayer<T1, T2>>;

}  // namespace lightseq
