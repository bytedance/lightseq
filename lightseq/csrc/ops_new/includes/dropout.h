#pragma once
#include "declaration.h"
#include "node.h"

namespace lightseq {

// after attention softmax
template <typename T1, typename T2>
class DropoutOp : public Operator {
 private:
  float ratio;
  int _max_ele_num;
  int _count;
  bool _is_skip;

  TensorPtr _mask;
  Variable* _result;

 public:
  float RATIO() const { return _context_ptr->is_training() ? ratio : 0.0; }

  DropoutOp(float r, int max_ele_num)
      : Operator("Dropout"), ratio(r), _max_ele_num(max_ele_num) {
    _mask.reset(new Tensor("mask", g_dtype<uint8_t>(), max_ele_num));
  }

  virtual ~DropoutOp() {}

  Variable* operator()(Variable* inp);

  void before_forward(int count) { _result->set_shape({count}); }

  void forward() override;

  void before_backward(int count) { _count = count; }

  void backward() override;
};
}  // namespace lightseq
