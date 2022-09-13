#pragma once
#include "bias_act_dropout.h"
#include "bias_add_transform_20314.h"
#include "bias_dropout_residual.h"
#include "dropout.h"
#include "feed_forward.h"
#include "layer_normalize.h"
#include "softmax.h"
#include "strided_batch_gemm.h"
#include "transform_0213.h"
#include "launch_concat3_dim1.h"
#include "layer.h"

namespace lightseq {

template <class T1, class T2>
class DecSelfAttentionLayer : public Layer {
 private:
  // operators
  LayerNormalizeOp<T1, T2>* _attn_ln = nullptr;
  FeedForwardOp<T1, T2>* _qkv_linear = nullptr;
  BiasAddTrans20314<T1, T2>* _bias_add_transform_20314 = nullptr;
  StridedBatchGemmOp<T1, T2>* _attn_scores = nullptr;
  SoftmaxOp<T1, T2>* _softmax = nullptr;
  DropoutOp<T1, T2>* _attn_prob_dropout = nullptr;
  StridedBatchGemmOp<T1, T2>* _attn_context = nullptr;
  Transform0213<T1, T2>* _transform_0213 = nullptr;
  FeedForwardOp<T1, T2>* _attn_out_linear = nullptr;
  BiasDropoutResOp<T1, T2>* _attn_dropout = nullptr;
  LaunchConcat3Dim1<T1, T2>* _deal_cache_k = nullptr;
  LaunchConcat3Dim1<T1, T2>* _deal_cache_v = nullptr;

  // parameters
  Variable* _attn_qkvw;
  Variable* _attn_qkvb;
  Variable* _attn_ow;
  Variable* _attn_ob;
  Variable* _attn_nw;
  Variable* _attn_nb;

  // shape related
  int _batch_dim;
  int _batch_heads;
  int _batch_tokens;
  int _layer_id;
  int _max_batch_tokens;
  int _max_seq_len;
  int _hidden_size;
  int _heads;
  bool _pre_or_postLayerNorm;
  bool _is_post_ln;

 public:
  DecSelfAttentionLayer(int layer_id, int max_batch_tokens, int max_seq_len,
                        int hidden_size, int num_heads,
                        float attn_prob_dropout_ratio,
                        float hidden_output_dropout_ratio,
                        bool pre_or_postLayerNorm, bool mask_future_tokens,
                        bool is_post_ln = false);

  virtual ~DecSelfAttentionLayer() {}

  std::tuple<Variable*, Variable*, Variable*> operator()(Variable* inp,
                                                         Variable* cache_k,
                                                         Variable* cache_v);

  void before_forward(int batch_size, int trg_seq_len, int src_seq_len, int steps, bool predict);

  void before_backward();

  int load_para_and_grad(const T1* para_ptr, T2* grad_ptr);

  int load_params(const std::vector<const T1*>& para_vec, int offset);
};

template class DecSelfAttentionLayer<__half, __half>;
template class DecSelfAttentionLayer<float, float>;

template <class T1, class T2>
using DecSelfAttentionLayerPtr = std::shared_ptr<DecSelfAttentionLayer<T1, T2>>;

}  // namespace lightseq
