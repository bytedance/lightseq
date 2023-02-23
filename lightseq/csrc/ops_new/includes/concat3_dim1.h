#pragma once
#include "declaration.h"
#include "node.h"
#include "tuple"

namespace lightseq {

template <typename T1, typename T2>
class Concat3Dim1 : public Operator {
 private:
  bool _is_skip = false;
  bool _is_continuous_cache;

  size_t _mx_sz0;
  size_t _mx_sz1;
  size_t _mx_sz2;

  size_t _sz0;
  size_t _sz1_0;
  size_t _sz1_1;

  Variable* _new_cache;

 public:
  Concat3Dim1(size_t mx_sz0, size_t mx_sz1, size_t mx_sz2, bool is_continuous_cache)
      : Operator("Concat3Dim1"),
        _mx_sz0(mx_sz0),
        _mx_sz1(mx_sz1),
        _mx_sz2(mx_sz2),
        _is_continuous_cache(is_continuous_cache) {}

  virtual ~Concat3Dim1() {}

  Variable* operator()(Variable* inp, Variable* cache);

  void before_forward(size_t sz0, size_t sz1_0, size_t sz1_1, bool is_skip = false) {
    _sz0 = sz0, _sz1_0 = sz1_0, _sz1_1 = sz1_1, _is_skip = is_skip;
    if (_is_continuous_cache) {
      _new_cache->set_shape({_sz0, _sz1_0 + _sz1_1, _mx_sz2});
    }
  }

  void forward() override;

  void before_backward() {}

  void backward() override;
};
}  // namespace lightseq
