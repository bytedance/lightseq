#pragma once
#include "declaration.h"
#include "node.h"
#include "transformerKernels.h"


namespace lightseq {

template <typename T>
class BeamSearchTopkOp : public Operator {
private:
    int _max_batch_size;
    int _max_step;
    int _trg_vocab_size;
    int _length_norm;
    int _cur_step;
    int _step_token_num;
    int _max_thread_per_block;
    int _beam_size;
    float _diverse_lambda;
    int _end_id;

    size_t _cub_sort_buffer_bytes;
    int _host_can_num_batch;

 public:
  BeamSearchTopkOp(int max_batch_size, int max_step, int trg_vocab_size, int max_thread_per_block, int beam_size, int diverse_lambda, int end_id)
      : Operator("BeamSearchTopkOp"),
      _max_batch_size(max_batch_size),
      _max_step(max_step),
      _trg_vocab_size(trg_vocab_size),
      _max_thread_per_block(max_thread_per_block),
      _beam_size(beam_size),
      _diverse_lambda(diverse_lambda),
      _end_id(end_id),
      _cub_sort_buffer_bytes(max_batch_size * beam_size * trg_vocab_size * sizeof(T)){}

  ~BeamSearchTopkOp() {}

  std::tuple<Variable*, Variable*, Variable*> operator()(Variable* logits, Variable* logit_bias, Variable* seq_probs,
    Variable* seq_score, Variable* alive_seq);

  void forward() override;

  void before_forward(int length_norm, int cur_step, int step_token_num) { 
      _length_norm = length_norm, _cur_step = cur_step, _step_token_num = step_token_num; 
  }

  void backward() override {}

  void before_backward() {}

  int can_num_batch() { return _host_can_num_batch; }
};

}  // namespace lightseq
