#include "launch_gpt_emb.h"

namespace lightseq {

template <typename T>
std::tuple<Variable*, Variable*, Variable*> LaunchGptEmbOp<T>::operator()(
    Variable* inp_tokens, Variable* token_emb, Variable* pos_emb) {
  set_parents({inp_tokens, token_emb, pos_emb});

  size_t max_size = _max_batch_tokens * _hidden_dim;

  _result = new Variable("LaunchGptEmbOp_out", _max_batch_tokens * _hidden_dim,
                         g_dtype<T>());
  _pad_mask = new Variable("_pad_mask", _max_batch_tokens, g_dtype<T>());

  _left_pad_len = new Variable("_left_pad_len", _max_batch_size * _beam_size,
                               g_dtype<int>(), cuda::DataType::kNotSupported,
                               VariableType::RegressiveVariable);

  this->set_children({_result, _pad_mask, _left_pad_len});
  return std::make_tuple(_result, _pad_mask, _left_pad_len);
}

template <typename T>
void LaunchGptEmbOp<T>::forward() {
  int* inp_tokens = (int*)parent(0)->value();
  const T* token_emb = (const T*)parent(1)->value();
  const T* pos_emb = (const T*)parent(2)->value();

  T* output_ptr = (T*)child(0)->value();
  T* pad_mask_ptr = (T*)child(1)->value();
  int* left_pad_len_ptr = (int*)child(2)->value();

  if (!_context_ptr->is_built()) {
    return;
  }

#ifdef LIGHTSEQ_cuda
  cudaStream_t _stream = _context_ptr->get_stream();
  if (_offset == 0) {
    cudaMemsetAsync(left_pad_len_ptr, 0, _batch_size * _beam_size * sizeof(int),
                    _stream);
  }
  cuda::launch_gpt_embedding<T>(token_emb, pos_emb, inp_tokens, output_ptr,
                                pad_mask_ptr, left_pad_len_ptr, _batch_size,
                                _beam_size, _hidden_dim, _offset, _seq_len,
                                _max_step, _pad_id, _stream);
#endif
}

template class LaunchGptEmbOp<float>;
#ifdef LIGHTSEQ_cuda
template class LaunchGptEmbOp<__half>;
#endif
}  // namespace lightseq
