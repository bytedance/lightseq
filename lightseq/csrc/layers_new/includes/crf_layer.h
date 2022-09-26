#pragma once

#include "crf.h"
#include "layer.h"

namespace lightseq {

template <typename T>
class CRFLayer : public Layer {
 private:
  // operators
  CRFOP<T>* _crf_op = nullptr;

  // parameters
  Variable* _linear_b;
  Variable* _start_transition;
  Variable* _end_transition;
  Variable* _transition;

  // shape related
  int _num_tags;
  int _max_batch_tokens;
  int _max_batch_size;

  int _seq_len;
  int _batch_size;
  bool _forward_or_decode;    // true for forward, false for decode
  bool _output_decode_score;  // true for output decode score

 public:
  CRFLayer(int num_tags, int max_batch_tokens, int max_batch_size);

  virtual ~CRFLayer() {}

  Variable* operator()(Variable* emission, Variable* mask);

  void before_forward(int batch_size, int seq_len, bool forward_or_decode,
                      bool output_decode_score);

  void before_backward();

  int load_params(const std::vector<const T*>& para_vec, int offset);
};

template class CRFLayer<__half>;
template class CRFLayer<float>;

template <class T>
using CRFLayerPtr = std::shared_ptr<CRFLayer<T>>;
}  // namespace lightseq
