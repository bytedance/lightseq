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
input: [batch_size, seq_len, qkv_num, hidden_size]
  qkv_num = 1 or 3, 1 for enc-dec cross attn, 3 for other attn
bias: [hidden_size]
query: [batch_size, nhead, seq_len, head_dim]
key: [batch_size, nhead, seq_len, head_dim]
value: [batch_size, nhead, seq_len, head_dim]

let's explain the SplitHeadOp by PyTorch:
input = input + bias
if qkv_num == 3:
  q, k, v = input.split(1, dim=2)
if qkv_num == 1:
  q = input
query = q.squeeze().reshape((batch_size, seq_len,
  nhead, head_dim)).permute(0, 2, 1, 3)
if qkv_num == 3:
  key = k.squeeze().reshape((batch_size, seq_len,
    nhead, head_dim)).permute(0, 2, 1, 3)
  value = v.squeeze().reshape((batch_size, seq_len,
    nhead, head_dim)).permute(0, 2, 1, 3)

*/
template <typename T1, typename T2>
class SplitHeadOp : public Operator {
 private:
  // const after init
  int _max_batch_tokens;  // batch_size * seq_len
  int _nhead;
  int _hidden_size;
  int _head_dim;
  int _qkv_num;

  // change every batch
  int _batch_size;
  int _seq_len;

 public:
  SplitHeadOp(int max_batch_tokens, int num_heads, int hidden_size,
              int qkv_num = 3)
      : Operator("SplitHeadOp"),
        _max_batch_tokens(max_batch_tokens),
        _nhead(num_heads),
        _hidden_size(hidden_size),
        _head_dim(hidden_size / num_heads),
        _qkv_num(qkv_num) {
    if (qkv_num != 1 && qkv_num != 3) {
      throw std::runtime_error("qkv_num should be 1 or 3.");
    }
  }

  virtual ~SplitHeadOp() {}

  std::tuple<Variable*, Variable*, Variable*> operator()(Variable* inp,
                                                         Variable* bias);

  void before_forward(int batch_size, int seq_len) {
    _batch_size = batch_size;
    _seq_len = seq_len;
  }

  void forward() override;

  void before_backward(int batch_size, int seq_len) {
    _batch_size = batch_size;
    _seq_len = seq_len;
  }

  void backward() override;
};

/*
@brief: SplitHeadWithBeamOp
add bias to input,
and split it into query, key, value

@param
input: [query_beam, batch_size, q_len, 3, hidden_size]
bias: [hidden_size]
query: [query_beam, batch_size, nhead, q_len, head_dim]
key: [beam_size, batch_size, nhead, cache_len, head_dim]
value: [beam_size, batch_size, nhead, cache_len, head_dim]
query_beam=beam_size if step>0 else 1

let's explain the SplitHeadOp by PyTorch:
input = input + bias
q, k, v = input.split(3, dim=2)
lambda func = x, size: x.squeeze().reshape(
  (size, batch_size, q_len, nhead, head_dim)).permute(
  (0, 1, 3, 2, 4))
query_beam=beam_size if step>0 else 1
query = func(q, query_beam)
k = func(k, query_beam)
v = func(v, query_beam)
if step == 0:
  k = k.repeat((beam_size, 1, 1, 1, 1))
  v = v.repeat((beam_size, 1, 1, 1, 1))
key[:, :, :, step:step+q_len, :] = k
value[:, :, :, step:step+q_len, :] = v
*/
template <typename T1, typename T2>
class SplitHeadWithBeamOp : public Operator {
 private:
  // const after init
  int _max_batch_tokens;  // batch_size * beam_size * q_len
  int _nhead;
  int _hidden_size;
  int _head_dim;
  int _beam_size;
  int _cache_len;

  // change every batch
  int _batch_size;
  int _q_len;
  int _step;

 public:
  SplitHeadWithBeamOp(int max_batch_tokens, int num_heads, int hidden_size,
                      int beam_size, int cache_len)
      : Operator("SplitHeadWithBeamOp"),
        _max_batch_tokens(max_batch_tokens),
        _nhead(num_heads),
        _hidden_size(hidden_size),
        _head_dim(hidden_size / num_heads),
        _beam_size(beam_size),
        _cache_len(cache_len) {}

  virtual ~SplitHeadWithBeamOp() {}

  Variable* operator()(Variable* inp, Variable* bias, Variable* cache_k,
                       Variable* cache_v);

  void before_forward(int batch_size, int q_len, int step) {
    _batch_size = batch_size;
    _q_len = q_len;
    _step = step;
    if (_step + _q_len >= _cache_len) {
      throw std::runtime_error("Exceed cache len.");
    }
  }

  void forward() override;

  void before_backward() {
    throw std::runtime_error("SplitHeadWithBeamOp does not have bw.");
  }

  void backward() {
    throw std::runtime_error("SplitHeadWithBeamOp does not have bw.");
  }
};

}  // namespace lightseq
