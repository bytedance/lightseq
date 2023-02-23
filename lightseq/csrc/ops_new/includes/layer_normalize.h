#pragma once
#include "declaration.h"
#include "node.h"

namespace lightseq {

template <class T1, class T2>
class LayerNormalizeOp : public Operator {
 private:
  size_t _max_batch_tokens;
  size_t _hidden_dim;
  size_t _batch_tokens;

  bool _use_mean;

  TensorPtr means_;
  TensorPtr vars_;

  Variable* _result;

 public:
  LayerNormalizeOp(size_t max_batch_tokens, size_t hidden_dim,
                   bool use_mean = false)
      : Operator("LayerNormalizeOp"),
        _max_batch_tokens(max_batch_tokens),
        _hidden_dim(hidden_dim),
        _use_mean(use_mean) {
    vars_.reset(new Tensor("vars", g_dtype<T1>(), max_batch_tokens));
    if (use_mean)
      means_.reset(new Tensor("means", g_dtype<T1>(), max_batch_tokens));
  }

  Variable* operator()(Variable* inp, Variable* gamma, Variable* betta);

  virtual ~LayerNormalizeOp();

  void before_forward(int batch_size, int seq_len);

  void forward() override;

  void backward() override;
};

}  // namespace lightseq
