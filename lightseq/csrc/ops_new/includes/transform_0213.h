#pragma once
#include "declaration.h"
#include "node.h"

namespace lightseq {

// [sz0, sz1, sz2, sz3] -> [sz0, sz2, sz1, sz3]
template <typename T1, typename T2>
class Transform0213OP : public Operator {
 private:
  int _max_numel;
  int _sz0;
  int _sz1;
  int _sz2;
  int _sz3;

 public:
  Transform0213OP(int max_numel)
      : Operator("Transform0213"), _max_numel(max_numel) {}

  virtual ~Transform0213OP() {}

  Variable* operator()(Variable* inp);

  void before_forward(int sz0, int sz1, int sz2, int sz3) {
    _sz0 = sz0, _sz1 = sz1, _sz2 = sz2, _sz3 = sz3;
  }

  void forward() override;

  void before_backward(int sz0, int sz1, int sz2, int sz3) {
    _sz0 = sz0, _sz1 = sz1, _sz2 = sz2, _sz3 = sz3;
  }

  void backward() override;
};
}  // namespace lightseq
