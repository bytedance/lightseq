#pragma once
#include "declaration.h"
#include "node.h"
#include "kernels.h"

namespace lightseq {

// transform 0213, execute after qkv_linear
template <typename T1, typename T2>
class Transform0213 : public Operator {
 private:
  int _max_batch_tokens;
  int _batch;
  int _seq_len;
  int _heads;
  int _hidden_size;

 public:
  Transform0213(int max_batch_tokens, int heads, int hidden_size)
      : Operator("Transform0213"),
        _max_batch_tokens(max_batch_tokens),
        _heads(heads),
        _hidden_size(hidden_size) {}

  virtual ~Transform0213() {}

  Variable* operator()(Variable* inp);

  void before_forward(int batch, int seq_len) {
    _batch = batch, _seq_len = seq_len;
  }

  void forward() override;

  void before_backward(int batch, int seq_len) {
    _batch = batch, _seq_len = seq_len;
  }

  void backward() override;
};
}  // namespace lightseq
