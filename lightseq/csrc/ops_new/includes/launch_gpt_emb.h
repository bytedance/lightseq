#pragma once
#include "declaration.h"
#include "node.h"
#include "tuple"

namespace lightseq {

// dropout inside ffn.
template <typename T>
class LaunchGptEmbOp : public Operator {
 private:
  size_t _max_batch_tokens;
  int _pad_id;
  size_t _hidden_dim;

  size_t _batch_size;
  size_t _seq_len;
  int _offset;

  Variable* _result;
  Variable* _result_seq_len;

 public:
  LaunchGptEmbOp(size_t max_batch_tokens, int pad_id, size_t hidden_dim)
      : Operator("LaunchGptEmbOp"),
        _max_batch_tokens(max_batch_tokens),
        _pad_id(pad_id),
        _hidden_dim(hidden_dim) {}

  virtual ~LaunchGptEmbOp() {}

  std::tuple<Variable*, Variable*> operator()(Variable* inp_tokens,
                                              Variable* token_emb,
                                              Variable* pos_emb);

  void before_forward(size_t batch_size, size_t seq_len, int offset) {
    _batch_size = batch_size, _seq_len = seq_len, _offset = offset;
    _result->set_shape({batch_size * seq_len, _hidden_dim});
    _result_seq_len->set_shape({batch_size});
  }

  void forward() override;

  void backward() override {
    printf("ERROR! LaunchGptEmbOp can't cal backward()\n");
    exit(-1);
  }
};
}  // namespace lightseq
