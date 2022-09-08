#pragma once
#include "declaration.h"
#include "node.h"
#include "kernels.h"
#include "tuple"

namespace lightseq {

template <typename T1, typename T2>
class LaunchConcat3Dim1 : public Operator {
 private:
  int _max_batchs;
  int _max_steps;
  int _seq_len;
  int _heads;
  int _hidden_size;
  int _batchs;
  int _steps;
  bool _predict;

 public:
  LaunchConcat3Dim1(int heads, int hidden_size)
      : Operator("LaunchConcat3Dim1"),
        _heads(heads),
        _hidden_size(hidden_size) {}

  virtual ~LaunchConcat3Dim1() {}

  Variable* operator()(Variable* inp, Variable* cache);

  void before_forward(int batchs, int seq_len, int steps, bool predict) {
    _batchs = batchs, _seq_len = seq_len, _steps = steps, _predict = predict;
  }

  void forward() override;

  void before_backward() {}

  void backward() override;
};
}  // namespace lightseq
