#pragma once
#include "declaration.h"
#include "node.h"
#include "kernels.h"
#include "tuple"

namespace lightseq {

template <typename T1, typename T2>
class Concat3Dim1 : public Operator {
 private:
  int _max_batchs;
  int _max_steps;
  int _seq_len;
  int _heads;
  int _hidden_size;
  int _batchs;
  int _steps;
  bool _is_skip;

 public:
  Concat3Dim1(int heads, int hidden_size, int max_steps = 0)
      : Operator("Concat3Dim1"),
        _heads(heads),
        _max_steps(max_steps),
        _hidden_size(hidden_size) {}

  virtual ~Concat3Dim1() {}

  Variable* operator()(Variable* inp, Variable* cache);

  void before_forward(int batchs, int seq_len, int steps,
                      bool is_skip = false) {

    _batchs = batchs, _seq_len = seq_len, _steps = steps, _is_skip = is_skip;
  }

  void forward() override;

  void before_backward() {}

  void backward() override;
};
}  // namespace lightseq
