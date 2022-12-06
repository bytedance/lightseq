#include "launch_enc_emb.h"

namespace lightseq {

template <typename T>
std::tuple<Variable*, Variable*> LaunchEncEmbOp<T>::operator()(
    Variable* inp_tokens, Variable* token_emb, Variable* pos_emb,
    Variable* lang_emb, Variable* lang_id) {
  size_t max_size = _max_batch_tokens * _hidden_dim;
  Variable* result = new Variable("LaunchEncEmbOp_out", max_size * sizeof(T));
  Variable* pad_mask =
      new Variable("pad_mask", _max_batch_tokens * sizeof(T));
  set_parents({inp_tokens, token_emb, pos_emb, lang_emb, lang_id});
  this->set_children({result, pad_mask});
  return std::make_tuple(result, pad_mask);
}

template <typename T>
void LaunchEncEmbOp<T>::forward() {
  cudaStream_t _stream = _context_ptr->get_stream();

  int* inp_tokens = (int*)parent(0)->value();
  const T* token_emb = (const T*)parent(1)->value();
  const T* pos_emb = (const T*)parent(2)->value();
  T* lang_emb = (T*)parent(3)->value();
  int* lang_id = (int*)parent(4)->value();

  T* output_ptr = (T*)child(0)->value();
  T* pad_mask = (T*)child(1)->value();

  if (!_context_ptr->is_built()) {
    return;
  }

  cuda::launch_enc_emb<T>(token_emb, pos_emb, inp_tokens, output_ptr, pad_mask,
                          _pad_id, _batch_size, _seq_len, _hidden_dim, _stream,
                          lang_emb, lang_id, _multilg_type);
}

template class LaunchEncEmbOp<float>;
template class LaunchEncEmbOp<__half>;

}  // namespace lightseq
