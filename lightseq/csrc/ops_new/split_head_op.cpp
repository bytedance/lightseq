#include "split_head_op.h"

namespace lightseq {

// without cache
template <typename T1, typename T2>
std::vector<Variable*> SplitHeadOp<T1, T2>::operator()(Variable* inp,
                                                       Variable* bias) {
  if (_cache_sz > 0) {
    throw std::runtime_error("Call the wrong version, should provide cache.");
  }
  set_parents({inp, bias});
  size_t trans_size = _max_query_tokens * _hidden_size;
  Variable* query = new Variable("splited_query", trans_size * sizeof(T1));
  if (_qkv_num == 1) {
    this->set_children({query});
    return {query};
  }
  Variable* key = new Variable("splited_key", trans_size * sizeof(T1));
  Variable* value = new Variable("splited_value", trans_size * sizeof(T1));
  this->set_children({query, key, value});
  return {query, key, value};
}

// with cache
template <typename T1, typename T2>
Variable* SplitHeadOp<T1, T2>::operator()(Variable* inp, Variable* bias,
                                          Variable* key, Variable* value) {
  if (_cache_sz == 0) {
    throw std::runtime_error(
        "Call the wrong version, should not provided cache.");
  }
  if (_qkv_num != 3) {
    throw std::runtime_error("qkv_num shoule be 3 when with cache.");
  }
  set_parents({inp, bias, key, value});
  size_t trans_size = _max_query_tokens * _hidden_size;
  Variable* query = new Variable("splited_query", trans_size * sizeof(T1));
  this->set_children({query});
  return query;
}

template <typename T1, typename T2>
void SplitHeadOp<T1, T2>::forward() {
  cudaStream_t _stream = _context_ptr->get_stream();

  T1* inp_ptr = (T1*)parent(0)->value();
  T1* bias_ptr = (T1*)parent(1)->value();

  T1* q_ptr = (T1*)child(0)->value();
  T1* k_ptr = nullptr;
  T1* v_ptr = nullptr;
  if (_qkv_num == 3) {
    if (_cache_sz == 0) {
      // without cache
      k_ptr = (T1*)child(1)->value();
      v_ptr = (T1*)child(2)->value();
    } else {
      // with cache
      k_ptr = (T1*)parent(2)->value();
      v_ptr = (T1*)parent(3)->value();
    }
  }

  if (!_context_ptr->is_built()) {
    return;
  }

  int kv_len = (_cache_sz > 0) ? _cache_sz : _q_len;
  launch_split_head<T1>(inp_ptr, bias_ptr, q_ptr, k_ptr, v_ptr, _batch_size,
                        _hidden_size, _head_dim, _q_len, kv_len, _step,
                        _qkv_num, _stream);
}

template <typename T1, typename T2>
void SplitHeadOp<T1, T2>::backward() {
  return;
}

template class SplitHeadOp<float, float>;
template class SplitHeadOp<__half, __half>;
}  // namespace lightseq
