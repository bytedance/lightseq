#pragma once

#include "linear.h"
#include "act_elewise_product.h"
#include "layer.h"

namespace lightseq {

template <class T1, class T2>
class LlamaMLPLayer : public Layer {
 private:
  // operators
  LinearOp<T1, T2>* _up_linear = nullptr;
  LinearOp<T1, T2>* _down_linear = nullptr;
  LinearOp<T1, T2>* _gate_linear = nullptr;
  ActElewiseProductOp<T1, T2>* _act_product = nullptr;

  // parameters
  Variable* _up_linear_weight;
  Variable* _down_linear_weight;
  Variable* _gate_linear_weight;

  // shape related
  int _max_batch_tokens;
  size_t _hidden_dim;
  size_t _inner_dim;

 public:
  LlamaMLPLayer(int max_batch_tokens, int hidden_dim, int inner_dim);

  virtual ~LlamaMLPLayer() {}

  Variable* operator()(Variable* inp);

  void before_forward(int batch_size, int seq_len);

  int load_params(const std::vector<const T1*>& para_vec, int offset);
};

template class LlamaMLPLayer<float, float>;
#ifdef LIGHTSEQ_cuda
template class LlamaMLPLayer<__half, __half>;
#endif

template <class T1, class T2>
using LlamaMLPLayerPtr = std::shared_ptr<LlamaMLPLayer<T1, T2>>;

}  // namespace lightseq
