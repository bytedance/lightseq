#pragma once
#include "declaration.h"
#include "node.h"

namespace lightseq {

// dropout inside ffn.
template <typename T1, typename T2>
class BiasActDropoutOp : public Operator {
 private:
  float ratio;

  int _mx_cols;
  int _mx_rows;
  int _cols;
  int _rows;

  std::string _activation_fn;

  TensorPtr _mask;

 public:
  float RATIO() const { return _context_ptr->is_training() ? ratio : 0.0; }

  BiasActDropoutOp(float r, size_t max_ele_num, std::string activation_fn)
      : Operator("BiasActDropoutOp"),
        ratio(r),
        _activation_fn(activation_fn),
        _mx_rows(mx_rows),
        _mx_cols(mx_cols) {
    _mask.reset(new Tensor("_mask", g_dtype<T1>(), {_mx_rows, _mx_cols}));
  }

  virtual ~BiasActDropoutOp() {}

  Variable* operator()(Variable* inp, Variable* bias);

  void before_forward(int rows, int cols) { _rows = rows, _cols = cols; }

  void forward() override;

  void before_backward(int rows, int cols) { _rows = rows, _cols = cols; }

  void backward() override;
};
}  // namespace lightseq
