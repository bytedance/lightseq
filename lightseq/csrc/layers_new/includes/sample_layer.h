#pragma once

#include "beam_search_topk.h"
#include "layer.h"

namespace lightseq {

template <class T>
class SampleLayer : public Layer {
 private:
  // operators
  BeamSearchTopOp<T>* _beam_search = nullptr;

  // parameters
  Variable* _logit_bias;

 public:
  SampleLayer(int nshared_dec_layer, int max_batch_size, int max_step,
              int trg_vocab_size, int hidden_size, int max_thread_per_block,
              int beam_size, int diverse_lambda, int dim_per_head, int end_id,
              int head_num, float length_penalty);  // for beam_search

  virtual ~SampleLayer() {}

  std::tuple<Variable*, Variable*> operator()(
      Variable* logits, Variable* alive_seq, Variable* caches_k,
      Variable* caches_k_buf, Variable* caches_v, Variable* caches_v_buf);

  void before_forward(int batch_size, int cur_step);

  void before_backward();

  int load_params(const std::vector<const T*>& para_vec, int offset);

  bool is_stop() { return _beam_search->is_stop(); }
};

template class SampleLayer<__half>;
template class SampleLayer<float>;

template <typename T>
using SampleLayerPtr = std::shared_ptr<SampleLayer<T>>;

}  // namespace lightseq
