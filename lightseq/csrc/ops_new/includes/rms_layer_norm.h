#pragma once
#include "declaration.h"
#include "node.h"
#include "kernels.h"

namespace lightseq {

template <class T1, class T2>
class RMSLayerNormalizeOp : public Operator {
 private:
  int _max_batch_tokens;
  int _hidden_dim;
  int _batch_tokens;

  bool _use_mean;

 public:
  RMSLayerNormalizeOp(uint32_t max_batch_tokens, uint32_t hidden_dim)
      : Operator("RMSLayerNormalizeOp"),
        _max_batch_tokens(max_batch_tokens),
        _hidden_dim(hidden_dim){
  }

  Variable* operator()(Variable* inp, Variable* gamma, Variable* betta);

  virtual ~RMSLayerNormalizeOp();

  void before_forward(size_t batch_tokens);

  void forward() override;

  void before_backward(size_t batch_tokens);

  void backward() override;
};

}  // namespace lightseq
