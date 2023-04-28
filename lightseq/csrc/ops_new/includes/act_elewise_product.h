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

  Variable* _result;


 public:
  ActElewiseProductOp(size_t max_batch_tokens, size_t inner_size)
      : Operator("ActElewiseProductOp"),
        _max_batch_tokens(max_batch_tokens),
        _inner_size(inner_size) {}

  ~ActElewiseProductOp() {}

  Variable* operator()(Variable* inpA, Variable* inpB);

  void forward() override;

  void before_forward(size_t batch_tokens) {
    _batch_tokens = batch_tokens;
    _result->set_shape({batch_tokens, _output_size});
  }

  void backward() override;

  void before_backward() {}
};

}  // namespace lightseq
