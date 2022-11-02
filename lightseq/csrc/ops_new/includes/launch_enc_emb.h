#pragma once
#include "declaration.h"
#include "node.h"
#include "kernels.h"
#include "embKernels.h"

namespace lightseq {

// dropout inside ffn.
template <typename T>
class LaunchEncEmbOp : public Operator {
 private:
  int _max_batch_tokens;
  int _pad_id;
  int _hidden_dim;
  int _multilg_type;

  int _batch_size;
  int _seq_len;

 public:
  LaunchEncEmbOp(int max_batch_tokens, int pad_id, int hidden_dim,
                 int multilg_type)
      : Operator("LaunchEncEmbOp"),
        _max_batch_tokens(max_batch_tokens),
        _pad_id(pad_id),
        _hidden_dim(hidden_dim),
        _multilg_type(multilg_type) {}

  virtual ~LaunchEncEmbOp() {}

  std::tuple<Variable*, Variable*> operator()(Variable* inp_tokens, Variable* token_emb,
                       Variable* pos_emb, Variable* lang_emb, Variable* lang_id);

  void before_forward(int batch_size, int seq_len) {
    _batch_size = batch_size, _seq_len = seq_len;
  }

  void forward() override;

  void before_backward() {}

  void backward() override {
    printf("ERROR! LaunchEncEmbOp can't cal backward()\n");
    exit(-1);
  }
};
}  // namespace lightseq
