#include "launch_dec_emb_op.h"

namespace lightseq {

template <typename T>
Variable* LaunchDecEmbOp<T>::operator()(Variable* inp_tokens,
                                        Variable* token_emb, Variable* pos_emb,
                                        Variable* lang_emb, Variable* lang_id) {
  size_t max_size = _max_batch_tokens * _hidden_size * _beam_size; 

  Variable* result = new Variable("LaunchDecEmbOp_out", max_size * sizeof(T));

  set_parents({inp_tokens, token_emb, pos_emb, lang_emb, lang_id});

  this->set_children({result});

  return result;
}

template <typename T>
void LaunchDecEmbOp<T>::forward() {
  cudaStream_t _stream = _context_ptr->get_stream();

  int* inp_tokens = (int*)parent(0)->value();
  const T* token_emb = (const T*)parent(1)->value();
  const T* pos_emb = (const T* const)parent(2)->value();
  T* lang_emb = (T*)parent(3)->value();
  int* lang_id = (int*)parent(4)->value();

  T* output_ptr = (T*)child(0)->value();
  
  cuda::launch_dec_emb<T>(token_emb, pos_emb, inp_tokens, lang_emb, lang_id,
                          output_ptr, _batch_size, _beam_size, _hidden_size,
                          _trg_vocab_size, _cur_step, _max_step, _multilg_type,
                          _stream);
}

template class LaunchDecEmbOp<float>;
template class LaunchDecEmbOp<__half>;

}  // namespace lightseq
