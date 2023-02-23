#pragma once
#include "declaration.h"
#include "node.h"

namespace lightseq {

// transformer layer's postprocessing dropout, after attn or ffn module,
// before residual add.
template <typename T1, typename T2>
class BiasDropoutResOp : public Operator {
 private:
  float ratio;

  size_t _max_rows;
  size_t _max_cols;
  size_t _rows;
  size_t _cols;

  TensorPtr _mask;
  Variable* _result;

 public:
  float RATIO() const { return _context_ptr->is_training() ? ratio : 0.0; }

  BiasDropoutResOp(float r, size_t max_rows, size_t max_cols)
      : Operator("BiasDropoutResOp"),
        ratio(r),
        _max_rows(max_rows),
        _max_cols(max_cols) {
    _mask.reset(new Tensor("mask", g_dtype<uint8_t>(), _max_rows * _max_cols));
  }

  virtual ~BiasDropoutResOp() {}

  Variable* operator()(Variable* inp, Variable* bias, Variable* residual);

  void before_forward(size_t rows, size_t cols) {
    _rows = rows, _cols = cols;
    _result->set_shape({_rows, _cols});
  }

  void forward() override;

  void backward() override;
};
}  // namespace lightseq
