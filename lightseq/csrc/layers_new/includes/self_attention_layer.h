#pragma once
#include "bias_act_dropout.h"
#include "bias_add_transform_20314.h"
#include "bias_dropout_residual.h"
#include "dropout.h"
#include "feed_forward.h"
#include "normalize_layer.h"
#include "softmax.h"
#include "strided_batch_gemm.h"
#include "transform_0213.h"
#include "layer.h"

namespace lightseq {


template <class T1, class T2>
class SelfAttentionLayer : public Layer {
private:
  // operators 
  NormalizeLayerOp<T1, T2>*     _attn_ln = nullptr;
  FeedForwardOp<T1, T2>*        _qkv_linear = nullptr;
  BiasAddTrans20314<T1, T2>*    _bias_add_transform_20314 = nullptr;
  StridedBatchGemmOp<T1, T2>*   _attn_scores = nullptr;
  SoftmaxOp<T1, T2>*            _softmax = nullptr;
  DropoutOp<T1, T2>*            _attn_prob_dropout = nullptr;
  StridedBatchGemmOp<T1, T2>*   _attn_context = nullptr;
  Transform0213<T1, T2>*        _transform_0213 = nullptr;
  FeedForwardOp<T1, T2>*        _attn_out_linear = nullptr;
  BiasDropoutResOp<T1, T2>*     _attn_dropout = nullptr;

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
  
public:
  SelfAttentionLayer();

  virtual ~SelfAttentionLayer() {}

  Variable* operator()(Variable* inp, Variable* inp_mask);

  void before_forward(int batch_size, int seq_len);

  void before_backward();
};

template class SelfAttentionLayer<__half, __half>;
template class SelfAttentionLayer<float, float>;

template <class T1, class T2>
using SelfAttentionLayer = std::shared_ptr<SelfAttentionLayer<T1, T2>>;

} // namespace lightseq 