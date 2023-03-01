#pragma once
#include "declaration.h"
#include "node.h"

namespace lightseq {

// dropout inside ffn.
template <typename T>
class LaunchDecEmbOp : public Operator {
 private:
  size_t _max_batch_tokens;
  size_t _beam_size;
  size_t _hidden_size;
  size_t _trg_vocab_size;
  size_t _max_step;
  size_t _multilg_type;

  size_t _batch_size;
  int _cur_step;

  Variable* _result;

 public:
  LaunchDecEmbOp(int max_batch_tokens, size_t beam_size, size_t hidden_size,
                 size_t trg_vocab_size, size_t max_step, size_t multilg_type)
      : Operator("LaunchDecEmbOp"),
        _max_batch_tokens(max_batch_tokens),
        _beam_size(beam_size),
        _hidden_size(hidden_size),
        _trg_vocab_size(trg_vocab_size),
        _max_step(max_step),
        _multilg_type(multilg_type) {}

  virtual ~LaunchDecEmbOp() {}

  Variable* operator()(Variable* inp_tokens, Variable* token_emb,
                       Variable* pos_emb, Variable* lang_emb,
                       Variable* lang_id);

  void before_forward(size_t batch_size, int cur_step) {
    _batch_size = batch_size, _cur_step = cur_step;
    _result->set_shape({batch_size, size_t(cur_step + 1), _beam_size, _hidden_size});
  }

  void forward() override;

  void backward() override {
    printf("ERROR! LaunchDecEmbOp can't cal backward()\n");
    exit(-1);
  }
};
}  // namespace lightseq
