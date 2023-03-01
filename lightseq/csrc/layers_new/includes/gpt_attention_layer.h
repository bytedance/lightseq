#pragma once
#include "bias_act_dropout.h"
#include "bias_add_transform_20314.h"
#include "bias_dropout_residual.h"
#include "dropout.h"
#include "linear.h"
#include "layer_normalize.h"
#include "sdpa_layer.h"
#include "transform_0213.h"
#include "concat3_dim1.h"
#include "layer.h"
#include "split_head_op.h"

namespace lightseq {

template <class T1, class T2>
class GptAttentionLayer : public Layer {
 private:
  // operators
  LayerNormalizeOp<T1, T2>* _attn_ln = nullptr;
  LinearOp<T1, T2>* _qkv_linear = nullptr;
  SplitHeadOp<T1, T2>* _split_head = nullptr;
  SDPALayer<T1, T2>* _sdpa = nullptr;
  Transform0213OP<T1, T2>* _transform_0213 = nullptr;
  LinearOp<T1, T2>* _attn_out_linear = nullptr;
  BiasDropoutResOp<T1, T2>* _attn_dropout = nullptr;

  // parameters
  Variable* _attn_qkvw;
  Variable* _attn_qkvb;
  Variable* _attn_ow;
  Variable* _attn_ob;
  Variable* _attn_nw;
  Variable* _attn_nb;

  // shape related
  int _max_batch_tokens;
  int _max_seq_len;
  int _hidden_size;
  int _nhead;
  int _head_dim;
  bool _is_pre_ln;

  // tensor slice
  Variable* _cache_k;
  Variable* _cache_v;

 public:
  GptAttentionLayer(int max_batch_tokens, int max_seq_len, int hidden_size,
                    int num_heads, int beam_size, float attn_prob_dropout_ratio,
                    float hidden_output_dropout_ratio, bool is_pre_ln = true);

  virtual ~GptAttentionLayer() {}

  Variable* operator()(Variable* inp);

  void before_forward(int batch_size, int trg_seq_len, int steps);

  void before_backward();

  size_t load_para_and_grad(const T1* para_ptr, T2* grad_ptr);

  int load_params(const std::vector<const T1*>& para_vec, int offset);
};

template class GptAttentionLayer<__half, __half>;
template class GptAttentionLayer<float, float>;

template <class T1, class T2>
using GptAttentionLayerPtr = std::shared_ptr<GptAttentionLayer<T1, T2>>;

}  // namespace lightseq
