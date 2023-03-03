#include "gpt_layer.h"

namespace lightseq {

template <typename T1, typename T2>
GptLayer<T1, T2>::GptLayer(int layer_id, int max_batch_tokens, int max_seq_len,
                           int hidden_size, int num_heads,
                           int intermediate_size, float attn_prob_dropout_ratio,
                           float activation_dropout_ratio,
                           float hidden_output_dropout_ratio,
                           std::string activation_fn, bool mask_future_tokens,
                           int beam_size)
    : Layer("GptLayer"), _layer_id(layer_id) {
  _attn_layer.reset(new GptAttentionLayer<T1, T2>(
      max_batch_tokens, max_seq_len, hidden_size, num_heads, beam_size,
      attn_prob_dropout_ratio, hidden_output_dropout_ratio, true));

  _ffn_layer.reset(new FeedForwardLayer<T1, T2>(
      layer_id, max_batch_tokens, max_seq_len, hidden_size, num_heads,
      intermediate_size, activation_dropout_ratio, hidden_output_dropout_ratio,
      true, activation_fn));

  this->_context_ptr->exit_layer();  // necessary
}

template <typename T1, typename T2>
Variable* GptLayer<T1, T2>::operator()(Variable* inp) {
  set_inputs({inp});

  Variable* attn_out = (*_attn_layer)(inp);

  Variable* ffn_out = (*_ffn_layer)(attn_out);

  set_outputs({ffn_out});
  return ffn_out;
}

template <typename T1, typename T2>
size_t GptLayer<T1, T2>::load_para_and_grad(const T1* para_ptr,
                                            T2* grad_ptr) {  // for training
  size_t offset = 0;

  offset +=
      _attn_layer->load_para_and_grad(para_ptr + offset, grad_ptr + offset);

  offset +=
      _ffn_layer->load_para_and_grad(para_ptr + offset, grad_ptr + offset);

  return offset;
}

template <typename T1, typename T2>
int GptLayer<T1, T2>::load_params(const std::vector<const T1*>& para_vec,
                                  int offset) {  // for inference
  int size = 0;

  size += _attn_layer->load_params(para_vec, offset + size);

  size += _ffn_layer->load_params(para_vec, offset + size);

  return size;
}

}  // namespace lightseq
