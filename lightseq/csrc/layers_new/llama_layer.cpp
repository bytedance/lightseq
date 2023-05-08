#include "llama_layer.h"

namespace lightseq {

template <typename T1, typename T2>
LlamaLayer<T1, T2>::LlamaLayer(int max_batch_tokens, int max_seq_len, int hidden_size, int inner_dim,
            int num_heads, int beam_size)
    : Layer("LlamaLayer"){
  _attn_layer.reset(new LlamaAttentionLayer<T1, T2>(
      max_batch_tokens, max_seq_len, hidden_size, num_heads, beam_size));

  _mlp_layer.reset(new LlamaMLPLayer<T1, T2>(max_batch_tokens, hidden_size, inner_dim));

  this->_context_ptr->exit_layer();  // necessary
}

template <typename T1, typename T2>
Variable* LlamaLayer<T1, T2>::operator()(Variable* inp, Variable* cache_k,
                                       Variable* cache_v, Variable* pad_mask) {
  set_inputs({inp, cache_k, cache_v, pad_mask});

  Variable* attn_out = (*_attn_layer)(inp, cache_k, cache_v, pad_mask);

  Variable* ffn_out = (*_mlp_layer)(attn_out);

  set_outputs({ffn_out});
  return ffn_out;
}

template <typename T1, typename T2>
int LlamaLayer<T1, T2>::load_params(const std::vector<const T1*>& para_vec,
                                  int offset) {  // for inference
  int size = 0;

  size += _attn_layer->load_params(para_vec, offset + size);

  size += _mlp_layer->load_params(para_vec, offset + size);

  return size;
}

}  // namespace lightseq
