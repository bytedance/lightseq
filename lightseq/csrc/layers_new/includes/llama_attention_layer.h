#pragma once
#include "layer.h"
#include "linear.h"
#include "rms_layer_norm.h"
#include "fuse_rotary_position_qkv.h"
#include "sdpa_layer.h"
#include "transform_0213.h"
#include "fuse_add2_op.h"

namespace lightseq {

template <class T1, class T2>
class LlamaAttentionLayer : public Layer {
 private:
  // operators
  RMSLayerNormalizeOp<T1, T2>* _attn_ln = nullptr;
  LinearOp<T1, T2>* _qkv_linear = nullptr;
  RotaryPositionQk<T1, T2>* _fuse_rotary = nullptr;
  SDPALayer<T1, T2>* _sdpa = nullptr;
  Transform0213OP<T1, T2>* _transform_0213 = nullptr;
  LinearOp<T1, T2>* _attn_out_linear = nullptr;
  FuseAdd2Op<T1, T2>* _add_residual = nullptr;

  // parameters
  Variable* _norm_scale;
  Variable* _attn_qkvw;
  Variable* _attn_ow;

  // shape related
  size_t _max_batch_size;
  int _max_batch_tokens;
  int _max_seq_len;
  size_t _hidden_size;
  int _nhead;
  int _head_dim;

  // tensor slice
  Variable* _cache_k;
  Variable* _cache_v;

 public:
  LlamaAttentionLayer(int max_batch_tokens, int max_seq_len, int hidden_size,
                      int num_heads, int beam_size);

  virtual ~LlamaAttentionLayer() {}

  Variable* operator()(Variable* inp, Variable* cache_k, Variable* cache_v,
                       Variable* pad_mask);

  void before_forward(int batch_size, int trg_seq_len, int prompt_len);

  void before_backward();

  int load_params(const std::vector<const T1*>& para_vec, int offset);
};

template class LlamaAttentionLayer<float, float>;
#ifdef LIGHTSEQ_cuda
template class LlamaAttentionLayer<__half, __half>;
#endif

template <class T1, class T2>
using LlamaAttentionLayerPtr = std::shared_ptr<LlamaAttentionLayer<T1, T2>>;

}  // namespace lightseq
