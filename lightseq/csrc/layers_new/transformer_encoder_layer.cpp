#include "transformer_encoder_layer.h"

namespace lightseq {

template <class T1, class T2>
int TransformerEncoderLayerWeight::load_para_and_grad(
    const T1* para_ptr, T2* grad_ptr) {  // for training
  int offset = 0;

  offset +=
      _attn_layer_wt->load_para_and_grad(para_ptr + offset, grad_ptr + offset);

  offset +=
      _ffn_layer_wt->load_para_and_grad(para_ptr + offset, grad_ptr + offset);

  return offset;
}

template int TransformerEncoderLayerWeight::load_para_and_grad(
    const float* para_ptr, float* grad_ptr);
template int TransformerEncoderLayerWeight::load_para_and_grad(
    const __half* para_ptr, __half* grad_ptr);

template <typename T>
int TransformerEncoderLayerWeight::load_params(
    const std::vector<const T*>& para_vec) {  // for inference
  int offset = 0;

  offset += _attn_layer_wt->load_params(
      std::vector<const T*>(para_vec.begin() + offset, para_vec.end()));

  offset += _ffn_layer_wt->load_params(
      std::vector<const T*>(para_vec.begin() + offset, para_vec.end()));

  return offset;
}

template int TransformerEncoderLayerWeight::load_params<float>(
    const std::vector<const float*>& para_vec);
template int TransformerEncoderLayerWeight::load_params<__half>(
    const std::vector<const __half*>& para_vec);

template <typename T1, typename T2>
TransformerEncoderLayer<T1, T2>::TransformerEncoderLayer(
    int layer_id, int max_batch_tokens, int max_seq_len, int hidden_size,
    int num_heads, int intermediate_size, float attn_prob_dropout_ratio,
    float activation_dropout_ratio, float hidden_output_dropout_ratio,
    bool pre_or_postLayerNorm, std::string activation_fn,
    bool mask_future_tokens, TransformerEncoderLayerWeightPtr enc_layer_wt)
    : Layer("TransformerEncoderLayer") {
  _attn_layer.reset(new MultiheadAttentionLayer<T1, T2>(
      layer_id, max_batch_tokens, max_seq_len, hidden_size, num_heads,
      attn_prob_dropout_ratio, hidden_output_dropout_ratio,
      pre_or_postLayerNorm, mask_future_tokens, enc_layer_wt->_attn_layer_wt));

  _ffn_layer.reset(new FeedForwardLayer<T1, T2>(
      layer_id, max_batch_tokens, max_seq_len, hidden_size, num_heads,
      intermediate_size, activation_dropout_ratio, hidden_output_dropout_ratio,
      pre_or_postLayerNorm, activation_fn, enc_layer_wt->_ffn_layer_wt));

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
