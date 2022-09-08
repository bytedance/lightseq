#include "launch_enc_emb.h"

namespace lightseq {

template <typename T>
Variable* LaunchEncEmbOp<T>::operator()(Variable* inp_tokens,
                                        Variable* token_emb, Variable* pos_emb,
                                        Variable* pad_mask, Variable* lang_emb,
                                        Variable* lang_id) {
  size_t max_size = _max_batch_tokens * _hidden_dim;
  Variable* result =
      new Variable("LaunchEncEmbOp_out", max_size * sizeof(T), 0);
  this->set_parents(
      {inp_tokens, token_emb, pos_emb, pad_mask, lang_emb, lang_id});
  this->set_children({result});
  return result;
}

template <typename T>
void LaunchEncEmbOp<T>::forward() {
  _context_ptr->build();
  cudaStream_t _stream = _context_ptr->get_stream();

  int* inp_tokens = (int*)parent(0)->value();
  const T* token_emb = (const T*)parent(1)->value();
  const T* pos_emb = (const T* const)parent(2)->value();
  int* pad_mask = (int*)parent(3)->value();
  T* lang_emb = (T*)parent(4)->value();
  int* lang_id = (int*)parent(5)->value();

  T* output_ptr = (T*)child(0)->value();

  cuda::launch_enc_emb<T>(token_emb, pos_emb, inp_tokens, output_ptr, pad_mask,
                          _pad_id, _batch_size, _seq_len, _hidden_dim, _stream,
                          lang_emb, lang_id, _multilg_type);
}

template class LaunchEncEmbOp<float>;
template class LaunchEncEmbOp<__half>;

}  // namespace lightseq
