#pragma once
#include "declaration.h"
#include "node.h"

namespace lightseq {

template <typename T1, typename T2>
class FuseAdd2Op : public Operator {
 private:
  size_t _max_batch_tokens;
  size_t _batch_tokens;
  size_t _batch_size;
  size_t _seq_len;
  size_t _hidden_dim;

  Variable* _result;

 public:
  FuseAdd2Op(size_t max_batch_tokens, size_t hidden_dim)
      : Operator("FuseAdd2"),
        _max_batch_tokens(max_batch_tokens),
        _hidden_dim(hidden_dim) {}

  ~FuseAdd2Op() {}

  Variable* operator()(Variable* inpA, Variable* inpB);

  void forward() override;

  void before_forward(size_t batch_size, size_t seq_len) {
    _batch_size = batch_size;
    _seq_len = seq_len;
    _result->set_shape({batch_size, seq_len, _hidden_dim});
  }

  void backward() override {}
};

}  // namespace lightseq
