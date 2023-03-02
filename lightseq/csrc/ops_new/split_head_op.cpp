#include "split_head_op.h"

namespace lightseq {

template <typename T1, typename T2>
std::tuple<Variable*, Variable*, Variable*> SplitHeadOp<T1, T2>::operator()(
    Variable* inp, Variable* bias) {
  set_parents({inp, bias});
  size_t trans_size = _max_batch_tokens * _hidden_size;
  Variable* query =
      new Variable("splited_query", trans_size, g_dtype<T1>(), g_dtype<T2>());
  if (_qkv_num == 1) {
    this->set_children({query});
    return std::make_tuple(query, nullptr, nullptr);
  }
  Variable* key =
      new Variable("splited_key", trans_size, g_dtype<T1>(), g_dtype<T2>());
  Variable* value =
      new Variable("splited_value", trans_size, g_dtype<T1>(), g_dtype<T2>());
  this->set_children({query, key, value});
  return std::make_tuple(query, key, value);
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
    k_ptr = (T1*)child(1)->value();
    v_ptr = (T1*)child(2)->value();
  }

  if (!_context_ptr->is_built()) {
    return;
  }
#ifdef LIGHTSEQ_cuda
  cuda::launch_split_head<T1>(inp_ptr, bias_ptr, q_ptr, k_ptr, v_ptr,
                              _batch_size, _hidden_size, _head_dim, _seq_len,
                              _qkv_num, _stream);
#endif
}

template <typename T1, typename T2>
void SplitHeadOp<T1, T2>::backward() {
  return;
}

template class SplitHeadOp<float, float>;
template class SplitHeadOp<__half, __half>;

template <typename T1, typename T2>
Variable* SplitHeadWithBeamOp<T1, T2>::operator()(Variable* inp, Variable* bias,
                                                  Variable* cache_k,
                                                  Variable* cache_v) {
  set_parents({inp, bias, cache_k, cache_v});
  size_t trans_size = _max_batch_tokens * _hidden_size;
  Variable* query =
      new Variable("splited_query", _max_batch_tokens * _hidden_size,
                   g_dtype<T1>(), g_dtype<T2>());
  this->set_children({query});
  return query;
}

template <typename T1, typename T2>
void SplitHeadWithBeamOp<T1, T2>::forward() {
  cudaStream_t _stream = _context_ptr->get_stream();

  T1* inp_ptr = (T1*)parent(0)->value();
  T1* bias_ptr = (T1*)parent(1)->value();
  T1* k_ptr = (T1*)parent(2)->value();
  T1* v_ptr = (T1*)parent(3)->value();

  T1* q_ptr = (T1*)child(0)->value();

  if (!_context_ptr->is_built()) {
    return;
  }
#ifdef LIGHTSEQ_cuda
  cuda::launch_split_head_with_beam<T1>(
      inp_ptr, bias_ptr, q_ptr, k_ptr, v_ptr, _batch_size, _hidden_size,
      _head_dim, _beam_size, _q_len, _step, _cache_len, _stream);
#endif
}

template class SplitHeadWithBeamOp<float, float>;
#ifdef LIGHTSEQ_cuda
template class SplitHeadWithBeamOp<__half, __half>;
#endif
}  // namespace lightseq
