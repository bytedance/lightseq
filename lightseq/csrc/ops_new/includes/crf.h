#pragma once
#include "declaration.h"
#include "node.h"

namespace lightseq {

// linear crf
template <typename T>
class CRFOP : public Operator {
 private:
  size_t _num_tags;
  size_t _seq_len;
  size_t _batch_size;
  size_t _max_batch_tokens;
  size_t _max_batch_size;

  bool _forward_or_decode;  // true for forward, false for decode
  bool _output_decode_score;
  TensorPtr _history;

  Variable* _best_tags;

 public:
  CRFOP(size_t max_batch_tokens, size_t max_batch_size, size_t num_tags);

  virtual ~CRFOP() {}

  Variable* operator()(Variable* start_transition, Variable* end_transition,
                       Variable* transition, Variable* emission, Variable* mask,
                       Variable* bias);

  void before_forward(size_t batch_size, size_t seq_len, bool forward_or_decode,
                      bool output_decode_score);

  void forward() override;

  void before_backward();

  void backward() override;
};

}  // namespace lightseq
