#pragma once
#include "declaration.h"
#include "node.h"
#include "tuple"

namespace lightseq {

// add bias and transform 20314, execute after qkv_linear
template <typename T1, typename T2>
class BiasAddTrans20314 : public Operator {
 private:
  int _max_batch_tokens;
  int _batch;
  int _seq_len;
  int _heads;
  int _hidden_size;
  int _trans_count;

  Variable* _res;

 public:
  BiasAddTrans20314(int max_batch_tokens, int heads, int hidden_size,
                    int trans_count)
      : Operator("BiasAddTrans20314"),
        _max_batch_tokens(max_batch_tokens),
        _heads(heads),
        _hidden_size(hidden_size),
        _trans_count(trans_count) {}

  virtual ~BiasAddTrans20314() {}

  Variable* operator()(Variable* inp, Variable* bias);

  void before_forward(int batch, int seq_len) {
    _batch = batch, _seq_len = seq_len;
    _res->set_shape({_trans_count, _batch * _seq_len, _hidden_size});
  }

  void forward() override;

  void backward() override;
};
}  // namespace lightseq
