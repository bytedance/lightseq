#include "launch_enc_emb.h"

namespace lightseq {

template <typename T>
Variable* LaunchEncEmbOp<T>::operator()(Variable* inp_tokens,
                                        Variable* token_emb, Variable* pos_emb,
                                        Variable* pad_mask, Variable* lang_emb,
                                        Variable* lang_id) {
  size_t max_size = _max_batch_tokens * _hidden_dim;
  Variable* result =
      new Variable(this->_name + "/out", max_size * sizeof(T), 0);
  this->set_parents(
      {inp_tokens, token_emb, pos_emb, pad_mask, lang_emb, lang_id});
  this->set_children({result});
  return result;
}

template <typename T>
void LaunchEncEmbOp<T>::forward() {
  cudaStream_t _stream = _context_ptr->get_stream();

  int* inp_tokens = (int*)parent(0)->value();
  T* token_emb = (T*)parent(1)->value();
  T* pos_emb = (T*)parent(2)->value();
  int* pad_mask = (int*)parent(3)->value();
  T* lang_emb = (T*)parent(4)->value();
  int* lang_id = (int*)parent(5)->value();

  T* output_ptr = (T*)child(0)->value();

  /* ---step2. encoder feedforward--- */
  launch_enc_emb<T>(token_emb, pos_emb, inp_tokens, output_ptr, pad_mask,
                    _pad_id, _batch_size, _seq_len, _hidden_dim, _stream,
                    lang_emb, lang_id, _multilg_type);
}

}  // namespace lightseq
