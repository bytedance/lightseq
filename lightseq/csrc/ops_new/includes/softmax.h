#pragma once
#include "declaration.h"
#include "node.h"
#include "kernels.h"
#include "cublas_wrappers.h"

namespace lightseq {

template <typename T1, typename T2>
class SoftmaxOp : public Operator {
 private:
  int _nhead;
  int _batch_tokens;
  int _from_len;
  int _to_len;
  bool _mask_future;

 public:
  SoftmaxOp(int nhead, bool mask_future)
      : Operator("Softmax"), _nhead(nhead), _mask_future(mask_future) {}

  virtual ~SoftmaxOp() {}

  Variable* operator()(Variable* inp, Variable* mask);

  void forward() override;

  void before_forward(int batch_tokens, int from_len, int to_len) {
    _batch_tokens = batch_tokens, _from_len = from_len, _to_len = to_len;
  }

  void backward() override;

  void before_backward(int batch_tokens, int from_len, int to_len) {
    _batch_tokens = batch_tokens, _from_len = from_len, _to_len = to_len;
  }
};

}  // namespace lightseq
