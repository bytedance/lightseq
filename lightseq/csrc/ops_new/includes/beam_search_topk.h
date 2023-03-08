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
  int _cur_pos;
  int _step_token_num;
  size_t _max_thread_per_block;
  size_t _beam_size;
  float _diverse_lambda;
  size_t _nshared_dec_layer;

  size_t _cub_sort_buffer_bytes;
  int _host_can_num_batch;
  size_t _batch_size;
  size_t _hidden_size;
  size_t _prompt_len;
  size_t _cache_size;
  int _step;
  int _end_id;
  size_t _dim_per_head;
  size_t _head_num;
  std::vector<float> _host_alive_seq_probs;
  std::vector<float> _host_length_norm;
  std::vector<int> _host_num_beam_can;

  Variable* _num_beam_can;
  Variable* _can_idx;
  Variable* _can_score;
  Variable* _seq_prob;
  Variable* _seq_score;
  Variable* _alive_seq_out;
  Variable* _caches_k_buf;
  Variable* _caches_v_buf;

 public:
  BeamSearchTopOp(size_t nshared_dec_layer, size_t max_batch_size,
                  size_t max_step, size_t trg_vocab_size, size_t hidden_size,
                  size_t max_thread_per_block, size_t beam_size,
                  size_t diverse_lambda, size_t dim_per_head, int end_id,
                  size_t head_num, float length_penalty);

  virtual ~BeamSearchTopOp() {}

  // output: out_token_ids, token_scores
  std::tuple<Variable*, Variable*> operator()(Variable* logits,
                                              Variable* logit_bias,
                                              Variable* alive_seq);

  void forward() override;

  void before_forward(size_t batch_size, size_t prompt_len, size_t step) {
    _batch_size = batch_size;
    _prompt_len = prompt_len;
    _step = step;
    _cur_pos = prompt_len + step - 1;
    _step_token_num = batch_size * _beam_size;

    _alive_seq_out->set_shape({_batch_size, _beam_size, size_t(_cur_pos + 1)});
    _seq_score->set_shape({_batch_size, _beam_size, size_t(_cur_pos + 1)});
  }

  void refresh_cache(Variable* caches_k, Variable* caches_v);

  void backward() override {}

  void before_backward() {}

  int is_stop() { return _host_can_num_batch == _step_token_num; }
};

}  // namespace lightseq
