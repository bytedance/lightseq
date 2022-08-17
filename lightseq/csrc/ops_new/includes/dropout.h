#pragma once
#include "declaration.h"
#include "node.h"
#include "kernels.h"

namespace lightseq {

// after attention softmax
template <typename T1, typename T2>
class DropoutOp : public Operator {
 private:
  float ratio;
  size_t _max_ele_num;
  int _count;

  TensorPtr _mask;

 public:
  float RATIO() const { return _context_ptr->is_training() ? ratio : 0.0; }

  DropoutOp(float r, size_t max_ele_num)
      : Operator("Dropout"), ratio(r), _max_ele_num(max_ele_num) {
    _mask.reset(new Tensor("_mask", max_ele_num * sizeof(uint8_t)));
  }

  virtual ~DropoutOp() {}

  Variable* operator()(Variable* inp);

  void before_forward(int count) { _count = count; }

  void forward() override;

  void before_backward(int count) { _count = count; }

  void backward() override;
};
}  // namespace lightseq
