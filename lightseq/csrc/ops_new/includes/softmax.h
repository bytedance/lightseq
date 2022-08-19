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

  int _max_batchs;
  int _max_from_len;
  int _max_to_len;

  int _batchs;
  int _from_len;
  int _to_len;
  bool _config_mask_future;
  bool _mask_future;

 public:
  SoftmaxOp(int max_batchs, int max_from_len, int max_to_len, int nhead,
            bool mask_future)
      : Operator("Softmax"),
        _max_batchs(max_batchs),
        _max_from_len(max_from_len),
        _max_to_len(max_to_len),
        _nhead(nhead),
        _config_mask_future(mask_future) {}

  virtual ~SoftmaxOp() {}

  Variable* operator()(Variable* inp, Variable* mask);

  void forward() override;

  void before_forward(int batchs, int from_len, int to_len,
                      bool mask_future = false) {
    _batchs = batchs;
    _from_len = from_len;
    _to_len = to_len;
    _mask_future = mask_future;
  }

  void backward() override;

  void before_backward(int batchs, int from_len, int to_len) {
    _batchs = batchs;
    _from_len = from_len;
    _to_len = to_len;
  }
};

}  // namespace lightseq
