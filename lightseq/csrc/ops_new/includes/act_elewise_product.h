#pragma once
#include "declaration.h"
#include "node.h"

namespace lightseq {

template <typename T1, typename T2>
class ActElewiseProductOp : public Operator {
 private:
  size_t _inner_size;
  size_t _max_batch_tokens;
  size_t _batch_tokens;
  size_t _batch_size;
  size_t _seq_len;

  Variable* _result;

 public:
  ActElewiseProductOp(size_t max_batch_tokens, size_t inner_size)
      : Operator("ActElewiseProductOp"),
        _max_batch_tokens(max_batch_tokens),
        _inner_size(inner_size) {}

  virtual ~ActElewiseProductOp() {}

  Variable* operator()(Variable* inp);

  void forward() override;

  void before_forward(size_t batch_size, size_t seq_len) {
    _batch_size = batch_size;
    _seq_len = seq_len;
    _batch_tokens = batch_size * seq_len;
    _result->set_shape({_batch_tokens, _inner_size});
  }

  void backward() override {}

  void before_backward() {}
};

}  // namespace lightseq
