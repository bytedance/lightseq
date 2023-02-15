#pragma once
#include "declaration.h"
#include "node.h"

namespace lightseq {
const float host_min_log_probability = -2000.f;

float host_length_norm(int length, float alpha) {
  if (alpha < 0.f) return 1.f / length;
  return std::pow((5.f + length) / 6.f, -alpha);
}

template <typename T>
class BeamSearchTopOp : public Operator {
 private:
  // inital
  int _max_batch_size;
  int _max_step;
  int _trg_vocab_size;
  int _length_norm;
  int _cur_step;
  int _step_token_num;
  int _max_thread_per_block;
  int _beam_size;
  float _diverse_lambda;
  int _nshared_dec_layer;

  size_t _cub_sort_buffer_bytes;
  int _host_can_num_batch;
  int _batch_size;
  int _cache_size;
  int _hidden_size;
  int _end_id;
  int _dim_per_head;
  int _head_num;
  std::vector<float> _host_alive_seq_probs;
  std::vector<float> _host_length_norm;

  Variable* _num_beam_can;
  Variable* _can_idx;
  Variable* _can_score;
  Variable* _seq_prob;
  Variable* _seq_score;
  Variable* _alive_seq_out;
  Variable* _caches_k_buf;
  Variable* _caches_v_buf;

 public:
  BeamSearchTopOp(int nshared_dec_layer, int max_batch_size, int max_step,
                  int trg_vocab_size, int hidden_size, int max_thread_per_block,
                  int beam_size, int diverse_lambda, int dim_per_head,
                  int end_id, int head_num, float length_penalty);

  ~BeamSearchTopOp() {}

  // output:
  std::tuple<Variable*, Variable*> operator()(Variable* logits,
                                              Variable* logit_bias,
                                              Variable* alive_seq,
                                              Variable* caches_k,
                                              Variable* caches_v);

  void forward() override;

  void before_forward(int batch_size, int cur_step) {
    _batch_size = batch_size;
    _cur_step = cur_step;
    _step_token_num = batch_size * _beam_size;

    _alive_seq_out->set_shape({_batch_size, _beam_size, _cur_step + 1});
    _seq_score->set_shape({_batch_size, _beam_size, _cur_step + 1});
  }

  void backward() override {}

  void before_backward() {}

  int is_stop() { return _host_can_num_batch == _step_token_num; }
};

}  // namespace lightseq
