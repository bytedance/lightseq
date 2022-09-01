#include "transformer_encoder_layer.h"

namespace lightseq {

template <typename T1, typename T2>
TransformerEncoderLayer<T1, T2>::TransformerEncoderLayer(
    int layer_id, int max_batch_tokens, int max_seq_len, int hidden_size,
    int num_heads, int intermediate_size, float attn_prob_dropout_ratio,
    float activation_dropout_ratio, float hidden_output_dropout_ratio,
    bool pre_or_postLayerNorm, std::string activation_fn,
    bool mask_future_tokens, bool is_post_ln)
    : Layer("TransformerEncoderLayer") {
  _attn_layer.reset(new MultiheadAttentionLayer<T1, T2>(
      layer_id, max_batch_tokens, max_seq_len, hidden_size, num_heads,
      attn_prob_dropout_ratio, hidden_output_dropout_ratio,
      pre_or_postLayerNorm, mask_future_tokens, is_post_ln));

  _ffn_layer.reset(new FeedForwardLayer<T1, T2>(
      layer_id, max_batch_tokens, max_seq_len, hidden_size, num_heads,
      intermediate_size, activation_dropout_ratio, hidden_output_dropout_ratio,
      pre_or_postLayerNorm, activation_fn, is_post_ln));

  this->_context_ptr->exit_layer();  // necessary
}

template <typename T1, typename T2>
Variable* TransformerEncoderLayer<T1, T2>::operator()(Variable* inp,
                                                      Variable* inp_mask) {
  LAYER_PRE_INPUTS({inp, inp_mask});

  Variable* attn_out = (*_attn_layer)(inp, inp_mask);

  Variable* ffn_out = (*_ffn_layer)(attn_out);

  LAYER_POST_OUTPUTS({ffn_out});
  return ffn_out;
}

template <typename T1, typename T2>
int TransformerEncoderLayer<T1, T2>::load_para_and_grad(
    const T1* para_ptr, T2* grad_ptr) {  // for training
  int offset = 0;

  offset +=
      _attn_layer->load_para_and_grad(para_ptr + offset, grad_ptr + offset);

  offset +=
      _ffn_layer->load_para_and_grad(para_ptr + offset, grad_ptr + offset);

  return offset;
}

template <typename T1, typename T2>
int TransformerEncoderLayer<T1, T2>::load_params(
    const std::vector<const T1*>& para_vec, int offset) {  // for inference
  int size = 0;

  size += _attn_layer->load_params(para_vec, offset + size);

  size += _ffn_layer->load_params(para_vec, offset + size);

  return size;
}

}  // namespace lightseq
