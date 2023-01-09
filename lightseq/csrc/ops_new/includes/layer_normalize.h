#pragma once
#include "declaration.h"
#include "node.h"
#include "headers.h"

namespace lightseq {

template <class T1, class T2>
class LayerNormalizeOp : public Operator {
 private:
  int _max_batch_tokens;
  int _hidden_dim;
  int _batch_tokens;

  bool _use_mean;

  TensorPtr means_;
  TensorPtr vars_;

 public:
  LayerNormalizeOp(uint32_t max_batch_tokens, uint32_t hidden_dim,
                   bool use_mean = false)
      : Operator("LayerNormalizeOp"),
        _max_batch_tokens(max_batch_tokens),
        _hidden_dim(hidden_dim),
        _use_mean(use_mean) {
    vars_.reset(new Tensor("vars", max_batch_tokens * sizeof(T1)));
    if (use_mean)
      means_.reset(new Tensor("means", max_batch_tokens * sizeof(T1)));
  }

  Variable* operator()(Variable* inp, Variable* gamma, Variable* betta);

  virtual ~LayerNormalizeOp();

  void before_forward(size_t batch_tokens);

  void forward() override;

  void before_backward(size_t batch_tokens);

  void backward() override;
};

}  // namespace lightseq
