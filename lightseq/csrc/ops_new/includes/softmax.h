#pragma once
#include "declaration.h"
#include "node.h"

namespace lightseq {

template <typename T1, typename T2>
class SoftmaxOp : public Operator {
 private:
  size_t _nhead;
  size_t _max_batch_tokens;
  size_t _max_seq_len;
  size_t _batchs;
  size_t _from_len;
  size_t _to_len;
  int _kv_size;

  bool _config_mask_future;
  bool _mask_future;

  Variable* _result;

 public:
  SoftmaxOp(size_t max_batch_tokens, size_t max_seq_len, size_t nhead,
            bool mask_future = false)
      : Operator("SoftmaxOp"),
        _max_batch_tokens(max_batch_tokens),
        _max_seq_len(max_seq_len),
        _nhead(nhead),
        _config_mask_future(mask_future) {}

  virtual ~SoftmaxOp() {}

  Variable* operator()(Variable* inp, Variable* mask = nullptr);

  void forward() override;

  void before_forward(size_t batchs, size_t from_len, size_t to_len, int kv_size = -1,
                      bool mask_future = false) {
    _batchs = batchs;
    _from_len = from_len;
    _to_len = to_len;
    _kv_size = (kv_size == -1 ? to_len : kv_size);
    _mask_future = mask_future;
    _result->set_shape({_batchs, _nhead, _from_len, _to_len});
  }

  void backward() override;
};

}  // namespace lightseq
