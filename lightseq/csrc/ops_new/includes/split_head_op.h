#pragma once
#include "declaration.h"
#include "node.h"
#include "kernels.h"
#include "tuple"

namespace lightseq {
/*
@brief: SplitHeadOp
add bias to input,
and split it into query, key, value

@param
input: [batch_size, q_len, qkv_num, hidden_size]
  qkv_num = 1 or 3, 1 for enc-dec cross attn, 3 for other attn
bias: [1, 1, qkv_num, hidden_size]
query: [batch_size, nhead, q_len, head_dim]
key: [batch_size, nhead, cache_sz, head_dim]
value: [batch_size, nhead, cache_sz, head_dim]

let's explain the SplitHeadOp by PyTorch:
input = input + bias
if qkv_num == 3:
  q, k, v = input.split(1, dim=2)
if qkv_num == 1:
  q = input

lambda func = x: x.squeeze().reshape((batch_size, seq_len,
  nhead, head_dim)).permute(0, 2, 1, 3)

query = func(q)
if qkv_num == 3:
  key[:,:,step:step+q_len,:] = func(k)
  value[:,:,step:step+q_len,:] = func(v)

*/
template <typename T1, typename T2>
class SplitHeadOp : public Operator {
 private:
  // const after init
  int _max_query_tokens;  // batch_size * q_len
  int _nhead;
  int _hidden_size;
  int _head_dim;
  int _qkv_num;
  int _cache_sz;

  // change every batch
  int _batch_size;
  int _q_len;
  int _step;

 public:
  SplitHeadOp(int max_query_tokens, int num_heads, int hidden_size,
              int qkv_num = 3, int cache_sz = 0)
      : Operator("SplitHeadOp"),
        _max_query_tokens(max_query_tokens),
        _nhead(num_heads),
        _hidden_size(hidden_size),
        _head_dim(hidden_size / num_heads),
        _cache_sz(cache_sz),
        _qkv_num(qkv_num) {
    if (qkv_num != 1 && qkv_num != 3) {
      throw std::runtime_error("qkv_num should be 1 or 3.");
    }
  }

  virtual ~SplitHeadOp() {}

  // without cache
  std::vector<Variable*> operator()(Variable* inp, Variable* bias);

  // with cache, return query
  Variable* operator()(Variable* inp, Variable* bias, Variable* key,
                       Variable* value);

  // without cache
  void before_forward(int batch_size, int q_len) {
    if (_cache_sz > 0) {
      throw std::runtime_error("should provide step when with cache.");
    }
    _batch_size = batch_size;
    _q_len = q_len;
    _step = 0;
  }

  // with cache
  void before_forward(int batch_size, int q_len, int step) {
    if (_cache_sz == 0) {
      throw std::runtime_error("should not provide step when without cache.");
    }
    _batch_size = batch_size;
    _q_len = q_len;
    _step = step;
    if (_step + _q_len > _cache_sz) {
      throw std::runtime_error("Exceed cache len.");
    }
  }

  void forward() override;

  void before_backward(int batch_size, int q_len) {
    _batch_size = batch_size;
    _q_len = q_len;
  }

  void backward() override;
};

}  // namespace lightseq
