#include "transformer_encoder_layer.h"

namespace lightseq {

template <typename T1, typename T2>
TransformerEncoderLayer<T1, T2>::TransformerEncoderLayer(
    int layer_id, int max_batch_tokens, int max_seq_len, int hidden_size,
    int num_heads, int intermediate_size, float attn_prob_dropout_ratio,
    float activation_dropout_ratio, float hidden_output_dropout_ratio,
    bool pre_or_postLayerNorm, std::string activation_fn,
    bool mask_future_tokens, const T1* para_ptr, T2* grad_ptr, int& offset)
    : Layer("TransformerEncoderLayer") {

  _attn_layer.reset(new SelfAttentionLayer<T1, T2>(
      layer_id, max_batch_tokens, max_seq_len, hidden_size, num_heads,
      attn_prob_dropout_ratio, hidden_output_dropout_ratio,
      pre_or_postLayerNorm, mask_future_tokens, para_ptr, grad_ptr, offset));


  _ffn_layer.reset(new FeedForwardLayer<T1, T2>(
      layer_id, max_batch_tokens, max_seq_len, hidden_size, num_heads,
      intermediate_size, activation_dropout_ratio, hidden_output_dropout_ratio,
      pre_or_postLayerNorm, activation_fn, para_ptr, grad_ptr, offset));

  this->_context_ptr->exit_layer();  // necessary
}

template <typename T1, typename T2>
Variable* TransformerEncoderLayer<T1, T2>::operator()(Variable* inp,
                                                      Variable* inp_mask) {
  this->set_inputs({inp, inp_mask});
  Variable* attn_out = (*_attn_layer)(inp, inp_mask);

  Variable* ffn_out = (*_ffn_layer)(attn_out);

  this->set_outputs({ffn_out});
  return ffn_out;
}

// template class TransformerEncoderLayer<float, float>;
// template class TransformerEncoderLayer<__half, __half>;

}  // namespace lightseq
