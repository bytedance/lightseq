#pragma once
#include "declaration.h"
#include "node.h"

namespace lightseq {

template <class T1, class T2>
class RMSLayerNormalizeOp : public Operator {
 private:
  size_t _max_batch_tokens;
  size_t _hidden_dim;
  size_t _batch_tokens;
  float _epsilon;

  bool _use_mean;
  bool _use_residual;

  TensorPtr _rms_vars;
  Variable* _result;
  Variable* _residual;

 public:
  RMSLayerNormalizeOp(size_t max_batch_tokens, size_t hidden_dim,
                      bool use_residual = true, float epsilon = 1e-6)
      : Operator("RMSLayerNormalizeOp"),
        _max_batch_tokens(max_batch_tokens),
        _hidden_dim(hidden_dim),
        _use_residual(use_residual),
        _epsilon(epsilon) {
    _rms_vars.reset(new Tensor("rms_vars", g_dtype<T1>(), max_batch_tokens));
  }

  std::tuple<Variable*, Variable*> operator()(Variable* inp, Variable* scale);

  virtual ~RMSLayerNormalizeOp();

  void before_forward(size_t batch_size, size_t seq_len);

  void forward() override;

  void backward() override {}
};

}  // namespace lightseq
