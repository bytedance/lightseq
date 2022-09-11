#include "crf.h"

namespace lightseq {

/*
start_transition: [num_tags]
end_transition: [num_tags]
transition: [num_tags, num_tags]
  transition[i, j] means the score of tag_j -> tag_i
emission: [batch_size, seq_len, num_tags]
mask: [batch_size, seq_len]
  0 for invalid token
best_score: [batch_size]
history: [batch_size, seq_len, num_tags]:
  i, j, k store the tag of i-th batch, j-th step when
  the tag of i-th batch, (j+1)-th step is k
best_tag: [batch_size, seq_len]
*/
template <typename T>
CRFOP<T>::CRFOP(int max_batch_tokens, int max_batch_size, int num_tags)
    : Operator("crf"),
      _max_batch_tokens(max_batch_tokens),
      _max_batch_size(max_batch_size),
      _num_tags(_num_tags) {
  _history =
      new Tensor(this->_name + "/history", _max_batch_tokens * sizeof(int));
}

template <typename T>
std::vector<Variable*> CRFOP<T>::operator()(Variable* start_transition,
                                            Variable* end_transition,
                                            Variable* transition,
                                            Variable* emission,
                                            Variable* mask) {
  Variable* best_tags =
      new Variable(this->_name + "/best_tags", _max_batch_tokens * sizeof(int));
  this->set_parents(
      {start_transition, end_transition, transition, emission, mask});
  if (!_output_decode_score) {
    this->set_children({best_tags});
    return {best_tags};
  }
  Variable* best_score = new Variable(this->_name + "/best_score",
                                      _max_batch_size * sizeof(float));
  this->set_children({best_tags, best_score});
  return {best_tags, best_score};
}

template <typename T>
void CRFOP<T>::before_forward(int batch_size, int seq_len,
                              bool forward_or_decode,
                              bool output_decode_score) {
  if (batch_size * seq_len > _max_batch_tokens) {
    throw std::runtime_error("batch_size * seq_len > _max_batch_tokens");
  }
  if (forward_or_decode) {
    throw std::runtime_error("CRF not support foward currently!");
  }
  _batch_size = batch_size;
  _seq_len = seq_len;
  _forward_or_decode = forward_or_decode;
  _output_decode_score = output_decode_score;
}

template <typename T>
void CRFOP<T>::forward() {
  cudaStream_t stream = _context_ptr->get_stream();

  const T* start_transition = (const T*)parent(0)->value();
  const T* end_transition = (const T*)parent(1)->value();
  const T* transition = (const T*)parent(2)->value();
  const T* emission = (const T*)parent(3)->value();
  const uint8_t* mask = (const uint8_t*)parent(4)->value();
  float* best_score =
      _output_decode_score ? (float*)child(1)->value() : nullptr;
  int* history = (int*)_history->tensor();
  int* best_tags = (int*)child(0)->value();

  launch_viterbi<T>(start_transition, end_transition, transition, emission,
                    mask, best_score, history, best_tags, _num_tags, _seq_len,
                    _batch_size, stream);
}

template <typename T>
void CRFOP<T>::before_backward() {
  throw std::runtime_error("CRF not support backward currently!");
}

template <typename T>
void CRFOP<T>::backward() {
  throw std::runtime_error("CRF not support backward currently!");
}

}  // namespace lightseq
