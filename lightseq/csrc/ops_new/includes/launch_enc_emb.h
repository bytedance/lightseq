#pragma once
#include "declaration.h"
#include "node.h"

namespace lightseq {

// dropout inside ffn.
template <typename T>
class LaunchEncEmbOp : public Operator {
 private:
  size_t _max_batch_tokens;
  int _pad_id;
  size_t _hidden_dim;
  size_t _multilg_type;

  size_t _batch_size;
  size_t _seq_len;

  Variable* _result;
  Variable* _pad_mask;

 public:
  LaunchEncEmbOp(size_t max_batch_tokens, int pad_id, size_t hidden_dim,
                 size_t multilg_type)
      : Operator("LaunchEncEmbOp"),
        _max_batch_tokens(max_batch_tokens),
        _pad_id(pad_id),
        _hidden_dim(hidden_dim),
        _multilg_type(multilg_type) {}

  virtual ~LaunchEncEmbOp() {}

  std::tuple<Variable*, Variable*> operator()(Variable* inp_tokens,
                                              Variable* token_emb,
                                              Variable* pos_emb,
                                              Variable* lang_emb,
                                              Variable* lang_id);

  void before_forward(size_t batch_size, size_t seq_len) {
    _batch_size = batch_size, _seq_len = seq_len;
  }

  void forward() override;

  void backward() override {
    printf("ERROR! LaunchEncEmbOp can't cal backward()\n");
    exit(-1);
  }
};
}  // namespace lightseq
