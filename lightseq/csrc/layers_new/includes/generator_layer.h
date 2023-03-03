#pragma once

#include "beam_search_topk.h"
#include "sampling.h"
#include "layer.h"

namespace lightseq {

template <class T>
class GeneratorLayer : public Layer {
 private:
  // operators
  BeamSearchTopOp<T>* _beam_search = nullptr;
  SamplingOp<T>* _sampling = nullptr;

  // parameters
  Variable* _logit_bias;
  size_t _trg_vocab_size;
  bool _has_logits_bias;

  GenerateMethod _generate_method;

 public:
  // this construct method is for beam_search generate method.
  GeneratorLayer(GenerateMethod gm, int max_batch_size, int max_step,
                 int trg_vocab_size, int hidden_size, int max_thread_per_block,
                 int beam_size = 0, float diverse_lambda = 0.,
                 int dim_per_head = 0, int end_id = 0, int head_num = 0,
                 float length_penalty = 0., int topk = 0, float topp = 0,
                 bool has_logits_bias = false);

  virtual ~GeneratorLayer() {}

  std::tuple<Variable*, Variable*> operator()(Variable* logits,
                                              Variable* alive_seq);

  void before_forward(int batch_size, int cur_step);

  int load_params(const std::vector<const T*>& para_vec, int offset);

  bool is_stop();
};

template class GeneratorLayer<float>;
#ifdef LIGHTSEQ_cuda
template class GeneratorLayer<__half>;
#endif

template <typename T>
using GeneratorLayerPtr = std::shared_ptr<GeneratorLayer<T>>;

}  // namespace lightseq
