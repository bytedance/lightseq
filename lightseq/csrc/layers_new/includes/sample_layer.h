#pragma once

#include "beam_search_topk.h"
#include "layer.h"

namespace lightseq {

template <class T>
class SampleLayer : public Layer {
 private:
  // operators
  BeamSearchTopOp<T>* beam_search = nullptr;

  // parameters

  
public:
  SampleLayer(int max_batch_size, int max_step, int trg_vocab_size,
              int max_thread_per_block, int beam_size,
              int diverse_lambda, int end_id); // for beam_search

  virtual ~SampleLayer() {}

  Variable* operator()(Variable* inp);

  void before_forward(int batch_size, int seq_len);

  void before_backward();

  int load_para_and_grad(const T1* para_ptr, T2* grad_ptr);

  int load_params(const std::vector<const T1*>& para_vec, int offset);
};

template class LinearLayer<__half, __half>;
template class LinearLayer<float, float>;

template <class T1, class T2>
using LinearLayerPtr = std::shared_ptr<LinearLayer<T1, T2>>;

}  // namespace lightseq
