#pragma once
#include "declaration.h"
#include "node.h"

namespace lightseq {

template <typename T>
class SamplingOp : public Operator {
 private:
  GenerateMethod _generate_method;
  int _max_batch_size;
  int _max_step;
  int _max_thread_per_block;
  int _trg_vocab_size;
  int _topk;
  float _topp;
  int _eos_id;
  bool _has_logits_bias;
  int* _p_d_unfinished;

  int _batch_size;
  int _seq_len;
  int _logits_seq_len;
  int _prompt_len;
  int _cur_step;

  int _h_unfinished;

#ifdef LIGHTSEQ_cuda
  curandState* _p_d_curandstate;  //[batch_size]
#endif

  Variable* _out_token_ids;
  Variable* _seq_score;

 public:
  SamplingOp(GenerateMethod gm, int max_batch_size, int max_step,
             int max_thread_per_block, int trg_vocab_size, int topk, float topp,
             int eos_id);

  // output: new_token_ids
  std::tuple<Variable*, Variable*> operator()(Variable* logits, Variable* logit_bias,
                       Variable* token_ids);

  void before_forward(int batch_size, int prompt_len, int cur_step, int logits_seq_len) {
    _batch_size = batch_size;
    _prompt_len = prompt_len;
    _cur_step = cur_step;
    _seq_len = prompt_len + cur_step;
    _logits_seq_len = logits_seq_len;
  }

  void forward() override;

  void backward() override {}

  bool is_stop() { return _h_unfinished == 0; }
};

}  // namespace lightseq
