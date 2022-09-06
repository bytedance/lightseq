#pragma once
#include "declaration.h"
#include "node.h"
#include "kernels.h"

namespace lightseq {

// dropout inside ffn.
template <typename T1, typename T2>
class BiasActDropoutOp : public Operator {
 private:
  float ratio;

  size_t _max_ele_num;
  int _count;
  int _cols, _rows;

  std::string _activation_fn;

  TensorPtr _mask;

 public:
  float RATIO() const { return _context_ptr->is_training() ? ratio : 0.0; }

  BiasActDropoutOp(float r, size_t max_ele_num, std::string activation_fn)
      : Operator("BiasActDropoutOp"),
        ratio(r),
        _activation_fn(activation_fn),
        _max_ele_num(max_ele_num) {
    _mask.reset(new Tensor(name() + "/_mask", max_ele_num * sizeof(uint8_t)));
  }

  virtual ~BiasActDropoutOp() {}

  Variable* operator()(Variable* inp, Variable* bias);

  void before_forward(int rows, int cols) { _rows = rows, _cols = cols; }

  void forward() override;

  void before_backward(int rows, int cols) { _rows = rows, _cols = cols; }

  void backward() override;
};
}  // namespace lightseq
