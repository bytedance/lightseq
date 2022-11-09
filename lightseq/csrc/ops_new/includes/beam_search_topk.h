#pragma once
#include "declaration.h"
#include "node.h"
#include "transformerKernels.h"

namespace lightseq {

template <typename T>
class BeamSearchTopOp : public Operator {
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

  Variable* alive_seq;
  Variable* alive_seq_buf;

 public:
  BeamSearchTopOp(int max_batch_size, int max_step, int trg_vocab_size,
                  int max_thread_per_block, int beam_size, int diverse_lambda,
                  int end_id);

  ~BeamSearchTopOp() {}

  std::tuple<Variable*, Variable*, Variable*> operator()(Variable* logits,
                                                         Variable* logit_bias,
                                                         Variable* seq_probs,
                                                         Variable* seq_score,
                                                         Variable* alive_seq);

  void forward() override;

  void before_forward(int length_norm, int cur_step, int step_token_num) {
    _length_norm = length_norm, _cur_step = cur_step,
    _step_token_num = step_token_num;
  }

  void backward() override {}

  void before_backward() {}

  int can_num_batch() { return _host_can_num_batch; }
};

}  // namespace lightseq
