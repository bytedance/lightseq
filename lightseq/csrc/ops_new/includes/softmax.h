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

  int _max_batch_tokens;
  int _max_seq_len;

  int _batchs;
  int _from_len;
  int _to_len;
  bool _config_mask_future;
  bool _mask_future;

 public:
  SoftmaxOp(int max_batch_tokens, int max_seq_len, int nhead,
            bool mask_future)
      : Operator("Softmax"),
        _max_batch_tokens(max_batch_tokens),
        _max_seq_len(max_seq_len),
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
