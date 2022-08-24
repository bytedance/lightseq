#pragma once
#include "declaration.h"
#include "node.h"
#include "kernels.h"

namespace lightseq {

// transformer layer's postprocessing dropout, after attn or ffn module,
// before residual add.
template <typename T1, typename T2>
class BiasDropoutResOp : public Operator {
 private:
  float ratio;

  size_t _max_ele_num;
  int _rows, _cols;

  TensorPtr _mask;

 public:
  float RATIO() const { return _context_ptr->is_training() ? ratio : 0.0; }

  BiasDropoutResOp(float r, size_t max_ele_num)
      : Operator("BiasDropoutResOp"), ratio(r), _max_ele_num(max_ele_num) {
    _mask.reset(
        new Tensor("BiasDropoutResOp/_mask", max_ele_num * sizeof(uint8_t)));
  }

  virtual ~BiasDropoutResOp() {}

  Variable* operator()(Variable* inp, Variable* bias, Variable* residual);

  void before_forward(int rows, int cols) { _rows = rows, _cols = cols; }

  void forward() override;

  void before_backward(int rows, int cols) { _rows = rows, _cols = cols; }

  void backward() override;
};
}  // namespace lightseq
