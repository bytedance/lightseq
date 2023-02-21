#pragma once
#include "declaration.h"
#include "node.h"

namespace lightseq {

// dropout inside ffn.
template <typename T>
class LaunchDecEmbOp : public Operator {
 private:
  int _max_batch_tokens;
  int _beam_size;
  int _hidden_size;
  int _trg_vocab_size;
  int _max_step;
  int _multilg_type;

  int _batch_size;
  int _cur_step;

  Variable* _result;

 public:
  LaunchDecEmbOp(int max_batch_tokens, int beam_size, int hidden_size,
                 int trg_vocab_size, int max_step, int multilg_type)
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

  void before_forward(int batch_size, int cur_step) {
    _batch_size = batch_size, _cur_step = cur_step;
    _result->set_shape({batch_size, cur_step + 1, _beam_size});
  }

  void forward() override;

  void backward() override {
    printf("ERROR! LaunchDecEmbOp can't cal backward()\n");
    exit(-1);
  }
};
}  // namespace lightseq
