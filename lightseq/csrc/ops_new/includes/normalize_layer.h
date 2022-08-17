#pragma once
#include "declaration.h"
#include "node.h"
#include "kernels.h"

namespace lightseq {

template <class T1, class T2>
class NormalizeLayerOp : public Operator {
 private:
  size_t _hidden_dim;
  size_t _max_batch_tokens;
  size_t _max_batch_dim;
  size_t _batch_tokens;

  bool _use_mean;

  TensorPtr means_;
  TensorPtr vars_;

 public:
  NormalizeLayerOp(uint32_t max_batch_tokens, uint32_t hidden_dim,
                   bool use_mean = false);

  Variable* operator()(Variable* inp, Variable* gamma, Variable* betta);

  virtual ~NormalizeLayerOp();

  void before_forward(size_t batch_tokens);

  void forward() override;

  void before_backward(size_t batch_tokens);

  void backward() override;
};

}  // namespace lightseq
