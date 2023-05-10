#include "llama_attention_layer.h"

namespace lightseq {

template <typename T1, typename T2>
LlamaAttentionLayer<T1, T2>::LlamaAttentionLayer(int max_batch_size,
                                                 int max_seq_len,
                                                 int hidden_size, int num_heads,
                                                 int beam_size)
    : Layer("LlamaAttentionLayer"),
      _max_batch_size(max_batch_size),
      _max_batch_tokens(max_batch_size * max_seq_len),
      _max_seq_len(max_seq_len),
      _hidden_size(hidden_size),
      _nhead(num_heads),
      _head_dim(hidden_size / num_heads) {
  // operators
  _attn_ln = new RMSLayerNormalizeOp<T1, T2>(_max_batch_tokens, hidden_size);
  _qkv_linear =
      new LinearOp<T1, T2>(_max_batch_tokens, 3 * hidden_size, hidden_size);
  _fuse_rotary = new RotaryPositionQk<T1, T2>(max_batch_size, max_seq_len,
                                              num_heads, _head_dim);

  _sdpa = new SDPALayer<T1, T2>(_max_batch_tokens, max_seq_len, _head_dim,
                                num_heads, 0.f);
  _transform_0213 =
      new Transform0213OP<T1, T2>(_max_batch_tokens * hidden_size);
  _attn_out_linear =
      new LinearOp<T1, T2>(_max_batch_tokens, hidden_size, hidden_size);
  // _add_residual = new FuseAdd2Op<T1, T2>(_max_batch_tokens, hidden_size);
  // parameters init
  _norm_scale = new Variable("_norm_scale", g_dtype<T1>(), g_dtype<T2>());
  _attn_qkvw = new Variable("_attn_qkvw", g_dtype<T1>(), g_dtype<T2>());
  _attn_ow = new Variable("_attn_ow", g_dtype<T1>(), g_dtype<T2>());

  this->_context_ptr->exit_layer();  // necessary
}

template <typename T1, typename T2>
Variable* LlamaAttentionLayer<T1, T2>::operator()(Variable* inp,
                                                  Variable* cache_k,
                                                  Variable* cache_v,
                                                  Variable* pad_mask) {
  set_inputs({inp, cache_k, cache_v, pad_mask});

  std::tuple<Variable*, Variable*> ln_out = (*_attn_ln)(inp, _norm_scale);
  Variable* qkv_out = (*_qkv_linear)(std::get<0>(ln_out), _attn_qkvw);

  Variable* q_out = (*_fuse_rotary)(qkv_out, cache_k, cache_v);

  // result of Scaled Dot Product Attention
  Variable* sdpa_res = (*_sdpa)(q_out, cache_k, cache_v, pad_mask);

  // [sz0, sz1, sz2, sz3] -> [sz0, sz2, sz1, sz3]
  Variable* transform_0213_out = (*_transform_0213)(sdpa_res);

  Variable* attn_linear =
      (*_attn_out_linear)(transform_0213_out, _attn_ow, std::get<1>(ln_out));

  // Variable* attn_out = (*_add_residual)(inp, attn_linear);

  set_outputs({attn_linear});
  return attn_linear;
}

template <typename T1, typename T2>
void LlamaAttentionLayer<T1, T2>::before_forward(int batch_size, int query_len,
                                                 int prompt_len) {
  // all token number in this batch
  int batch_tokens = batch_size * query_len;
  int attn_to_len = (prompt_len <= 0) ? query_len : prompt_len + 1;

  _attn_ln->before_forward(batch_size, query_len);

  _qkv_linear->before_forward(batch_tokens);

  _fuse_rotary->before_forward(batch_size, prompt_len, query_len);

  // mask future when training or (inference and prompt_len=0)
  _sdpa->before_forward(batch_size, query_len, attn_to_len, _max_seq_len,
                        prompt_len <= 0);

  _transform_0213->before_forward(batch_size, _nhead, query_len, _head_dim);

  _attn_out_linear->before_forward(batch_tokens);

  // _add_residual->before_forward(batch_size, query_len);
}

template <typename T1, typename T2>
void LlamaAttentionLayer<T1, T2>::before_backward() {}

template <typename T1, typename T2>
int LlamaAttentionLayer<T1, T2>::load_params(
    const std::vector<const T1*>& para_vec, int offset) {  // for inference
  int size = 0;
  _norm_scale->set_value((char*)para_vec[offset + size]), size++;
  _norm_scale->set_shape({_hidden_size});

  _attn_qkvw->set_value((char*)para_vec[offset + size]), size++;
  _attn_qkvw->set_shape({_hidden_size, 3 * _hidden_size});

  _attn_ow->set_value((char*)para_vec[offset + size]), size++;
  _attn_ow->set_shape({_hidden_size, _hidden_size});

  return size;
}

}  // namespace lightseq
