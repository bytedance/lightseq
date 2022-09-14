#include "crf_layer.h"

namespace lightseq {

template <typename T>
CRFLayer<T>::CRFLayer(int num_tags, int max_batch_tokens, int max_batch_size,
                      const T* start_transition, const T* end_transition,
                      const T* transition)
    : Layer("crflayer"),
      _num_tags(num_tags),
      _max_batch_tokens(max_batch_tokens),
      _max_batch_size(max_batch_size) {
  // operators node
  _crf_op = new CRFOP<T>(max_batch_tokens, max_batch_size, num_tags);
  // parameters node
  _start_transition = new Variable(this->_name + ":start_transition",
                                   (const char*)start_transition);
  _end_transition = new Variable(this->_name + ":end_transition",
                                 (const char*)end_transition);
  _transition =
      new Variable(this->_name + ":transition", (const char*)transition);
  this->_context_ptr->exit_layer();  // necessary
}

template <typename T>
std::vector<Variable*> CRFLayer<T>::operator()(Variable* emission,
                                               Variable* mask) {
  return (*_crf_op)(_start_transition, _end_transition, _transition, emission,
                    mask);
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

template class CRFLayer<__half>;

}  // namespace lightseq
