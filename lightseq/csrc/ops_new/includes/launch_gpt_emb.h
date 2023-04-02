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
  int _max_step;
  int _beam_size;
  int _offset;
  int _max_batch_size;

  Variable* _result;
  Variable* _pad_mask;
  Variable* _left_pad_len;

 public:
  LaunchGptEmbOp(size_t max_batch_tokens, int max_step, int max_batch_size, int beam_size,
                 int pad_id, size_t hidden_dim)
      : Operator("LaunchGptEmbOp"),
        _max_batch_tokens(max_batch_tokens),
        _max_batch_size(max_batch_size),
        _pad_id(pad_id),
        _max_step(max_step),
        _beam_size(beam_size),
        _hidden_dim(hidden_dim) {}

  virtual ~LaunchGptEmbOp() {}

  std::tuple<Variable*, Variable*, Variable*> operator()(Variable* inp_tokens,
                                              Variable* token_emb,
                                              Variable* pos_emb);

  void before_forward(size_t batch_size, size_t seq_len, int offset) {
    _batch_size = batch_size, _seq_len = seq_len, _offset = offset;
    _result->set_shape({batch_size * seq_len, _hidden_dim});
    _pad_mask->set_shape({batch_size, seq_len + offset});
  }

  void forward() override;

  void backward() override {
    printf("ERROR! LaunchGptEmbOp can't cal backward()\n");
    exit(-1);
  }
};
}  // namespace lightseq
