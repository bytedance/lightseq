#pragma once
#include "declaration.h"
#include "node.h"

namespace lightseq {
const float host_min_log_probability = -2000.f;

float host_length_norm_func(int length, float alpha);

template <typename T>
class BeamSearchTopOp : public Operator {
 private:
  // inital
  size_t _max_batch_size;
  size_t _max_step;
  size_t _trg_vocab_size;
  size_t _length_norm;
  int _cur_step;
  int _step_token_num;
  size_t _max_thread_per_block;
  size_t _beam_size;
  float _diverse_lambda;
  size_t _nshared_dec_layer;
  bool _with_start_id;

  size_t _cub_sort_buffer_bytes;
  int _host_can_num_batch;
  size_t _batch_size;
  size_t _hidden_size;
  int _end_id;
  size_t _dim_per_head;
  size_t _head_num;
  std::vector<float> _host_alive_seq_probs;
  std::vector<float> _host_length_norm;

  Variable* _num_beam_can;
  Variable* _can_idx;
  Variable* _can_score;
  Variable* _seq_prob;
  Variable* _seq_score;
  Variable* _alive_seq_out;

 public:
  BeamSearchTopOp(size_t max_batch_size, size_t max_step, size_t trg_vocab_size,
                  size_t hidden_size, size_t max_thread_per_block,
                  size_t beam_size, size_t diverse_lambda, size_t dim_per_head,
                  int end_id, size_t head_num, float length_penalty,
                  bool with_start_id = false);

  virtual ~BeamSearchTopOp() {}

  // output: out_token_ids, token_scores
  std::tuple<Variable*, Variable*> operator()(Variable* logits,
                                              Variable* logit_bias,
                                              Variable* alive_seq);

  void forward() override;

  void before_forward(size_t batch_size, size_t cur_step) {
    _batch_size = batch_size;
    _cur_step = cur_step;
    _step_token_num = batch_size * _beam_size;

    _alive_seq_out->set_shape({_batch_size, _beam_size, size_t(_cur_step + 1)});
    _seq_score->set_shape({_batch_size, _beam_size, size_t(_cur_step + 1)});
  }

  void backward() override {}

  void before_backward() {}

  int is_stop() { return _host_can_num_batch == _step_token_num; }
};

}  // namespace lightseq
