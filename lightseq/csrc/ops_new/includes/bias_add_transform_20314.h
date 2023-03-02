#pragma once
#include "declaration.h"
#include "node.h"
#include "tuple"

namespace lightseq {

// add bias and transform 20314, execute after qkv_linear
template <typename T1, typename T2>
class BiasAddTrans20314 : public Operator {
 private:
  size_t _max_batch_tokens;
  size_t _batch;
  size_t _seq_len;
  size_t _heads;
  size_t _hidden_size;
  size_t _trans_count;

  Variable* _res;

 public:
  BiasAddTrans20314(size_t max_batch_tokens, size_t heads, size_t hidden_size,
                    size_t trans_count)
      : Operator("BiasAddTrans20314"),
        _max_batch_tokens(max_batch_tokens),
        _heads(heads),
        _hidden_size(hidden_size),
        _trans_count(trans_count) {}

  virtual ~BiasAddTrans20314() {}

  Variable* operator()(Variable* inp, Variable* bias);

  void before_forward(size_t batch, size_t seq_len) {
    _batch = batch, _seq_len = seq_len;
    _res->set_shape(
        {_trans_count, _batch, _heads, _seq_len, _hidden_size / _heads});
  }

  void forward() override;

  void backward() override;
};
}  // namespace lightseq
