#include "crf_layer.h"

namespace lightseq {

template <typename T>
CRFLayer<T>::CRFLayer(int num_tags, int max_batch_tokens, int max_batch_size,
                      const T* start_transition, const T* end_transition,
                      const T* transition)
    : Layer("crflayer"),
      _num_tags(num_tags),
      _max_batch_tokens(max_batch_tokens),
      _max_batch_size(max_batch_size),

      // operators
      _crf_op(
          new CRFOP<T>(int max_batch_tokens, int max_batch_size, int num_tags)),
      _ff1(new FeedForwardOp<T1, T2>(max_batch_tokens, intermediate_size,
                                     hidden_size)),
      _ffn_activation_dropout(new BiasActDropoutOp<T1, T2>(
          activation_dropout_ratio, max_batch_tokens * intermediate_size,
          activation_fn)),
      _ff2(new FeedForwardOp<T1, T2>(max_batch_tokens, hidden_size,
                                     intermediate_size)),
      _ffn_dropout(new BiasDropoutResOp<T1, T2>(
          hidden_output_dropout_ratio, max_batch_tokens * hidden_size)) {
  // parameters node
  _inter_w = new Variable(this->_name + "_inter_w", (char*)(para_ptr + offset),
                          (char*)(grad_ptr + offset));
  offset += _hidden_size * _intermediate_size;
  _inter_b = new Variable(this->_name + "_inter_b", (char*)(para_ptr + offset),
                          (char*)(grad_ptr + offset));
  offset += _intermediate_size;

  _output_w =
      new Variable(this->_name + "_output_w", (char*)(para_ptr + offset),
                   (char*)(grad_ptr + offset));
  offset += _hidden_size * _intermediate_size;
  _output_b =
      new Variable(this->_name + "_output_b", (char*)(para_ptr + offset),
                   (char*)(grad_ptr + offset));
  offset += _hidden_size;

  _ffn_nw = new Variable(this->_name + "_ffn_nw", (char*)(para_ptr + offset),
                         (char*)(grad_ptr + offset));
  offset += _hidden_size;
  _ffn_nb = new Variable(this->_name + "_ffn_nb", (char*)(para_ptr + offset),
                         (char*)(grad_ptr + offset));
  offset += _hidden_size;

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

}  // namespace lightseq
