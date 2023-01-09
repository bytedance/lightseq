#pragma once
#include "declaration.h"
#include "node.h"
#include "headers.h"
#include "tuple"

namespace lightseq {

template <typename T1, typename T2>
class Concat3Dim1 : public Operator {
 private:
  int _mx_sz0;
  int _mx_sz1;
  int _mx_sz2;
  bool _is_skip = false;
  bool _is_continuous_cache;

  int _sz0;
  int _sz1_0;
  int _sz1_1;

 public:
  Concat3Dim1(int mx_sz0, int mx_sz1, int mx_sz2, bool is_continuous_cache)
      : Operator("Concat3Dim1"),
        _mx_sz0(mx_sz0),
        _mx_sz1(mx_sz1),
        _mx_sz2(mx_sz2),
        _is_continuous_cache(is_continuous_cache) {}

  virtual ~Concat3Dim1() {}

  Variable* operator()(Variable* inp, Variable* cache);

  void before_forward(int sz0, int sz1_0, int sz1_1, bool is_skip = false) {
    _sz0 = sz0, _sz1_0 = sz1_0, _sz1_1 = sz1_1, _is_skip = is_skip;
  }

  void forward() override;

  void before_backward() {}

  void backward() override;
};
}  // namespace lightseq
