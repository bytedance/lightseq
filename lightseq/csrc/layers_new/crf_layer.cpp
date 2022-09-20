#include "crf_layer.h"

namespace lightseq {

template <typename T>
CRFLayer<T>::CRFLayer(int num_tags, int max_batch_tokens, int max_batch_size)
    : Layer("CRFLayer"),
      _num_tags(num_tags),
      _max_batch_tokens(max_batch_tokens),
      _max_batch_size(max_batch_size) {
  // operators node
  _crf_op = new CRFOP<T>(max_batch_tokens, max_batch_size, num_tags);
  // parameters node
  _linear_b = new Variable("linear_b");
  _start_transition = new Variable("start_transition");
  _end_transition = new Variable("end_transition");
  _transition = new Variable("transition");

  this->_context_ptr->exit_layer();  // necessary
}

template <typename T>
Variable* CRFLayer<T>::operator()(Variable* emission, Variable* mask) {
  LAYER_PRE_INPUTS({emission, mask});
  Variable* crf_out = (*_crf_op)(_start_transition, _end_transition,
                                 _transition, emission, mask, _linear_b);
  LAYER_POST_OUTPUTS(crf_out);
  return crf_out;
}

template <typename T>
void CRFLayer<T>::before_forward(int batch_size, int seq_len,
                                 bool forward_or_decode,
                                 bool output_decode_score) {
  _crf_op->before_forward(batch_size, seq_len, forward_or_decode,
                          output_decode_score);
}

template <typename T>
void CRFLayer<T>::before_backward() {
  throw std::runtime_error("CRF not support backward currently!");
}

template <typename T>
int CRFLayer<T>::load_params(const std::vector<const T*>& para_vec,
                             int offset) {  // for inference
  int size = 0;
  _linear_b->set_value((char*)para_vec[offset + size]), size++;
  _start_transition->set_value((char*)para_vec[offset + size]), size++;
  _end_transition->set_value((char*)para_vec[offset + size]), size++;
  _transition->set_value((char*)para_vec[offset + size]), size++;

  return size;
}

}  // namespace lightseq
