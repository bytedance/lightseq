#pragma once
#include "declaration.h"
#include "node.h"
#include "kernels.h"

namespace lightseq {

// linear crf
template <typename T>
class CRFOP : public Operator {
 private:
  int _num_tags;
  int _seq_len;
  int _batch_size;
  int _max_batch_tokens;
  int _max_batch_size;

  bool _forward_or_decode;  // ture for forward, false for decode
  bool _output_decode_score;
  TensorPtr _history;

 public:
  CRFOP(int max_batch_tokens, int max_batch_size, int num_tags);

  virtual ~CRFOP() {}

  std::vector<Variable*> operator()(Variable* start_transition,
                                    Variable* end_transition,
                                    Variable* transition, Variable* emission,
                                    Variable* mask);

  void before_forward(int batch_size, int seq_len, bool forward_or_decode,
                      bool output_decode_score);

  void forward() override;

  void before_backward();

  void backward() override;
};

}  // namespace lightseq
