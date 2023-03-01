#pragma once
#include "bias_act_dropout.h"
#include "bias_add_transform_20314.h"
#include "bias_dropout_residual.h"
#include "dropout.h"
#include "linear.h"
#include "layer_normalize.h"
#include "softmax.h"
#include "strided_batch_gemm.h"
#include "transform_0213.h"
#include "concat3_dim1.h"
#include "layer.h"

namespace lightseq {

template <class T1, class T2>
class DecSelfAttentionLayer : public Layer {
 private:
  // operators
  LayerNormalizeOp<T1, T2>* _attn_ln = nullptr;
  LinearOp<T1, T2>* _qkv_linear = nullptr;
  BiasAddTrans20314<T1, T2>* _bias_add_transform_20314 = nullptr;
  StridedBatchGemmOp<T1, T2>* _attn_scores = nullptr;
  SoftmaxOp<T1, T2>* _softmax = nullptr;
  DropoutOp<T1, T2>* _attn_prob_dropout = nullptr;
  StridedBatchGemmOp<T1, T2>* _attn_context = nullptr;
  Transform0213OP<T1, T2>* _transform_0213 = nullptr;
  LinearOp<T1, T2>* _attn_out_linear = nullptr;
  BiasDropoutResOp<T1, T2>* _attn_dropout = nullptr;
  Concat3Dim1<T1, T2>* _concat_cache_k = nullptr;
  Concat3Dim1<T1, T2>* _concat_cache_v = nullptr;

  // parameters
  Variable* _attn_qkvw;
  Variable* _attn_qkvb;
  Variable* _attn_ow;
  Variable* _attn_ob;
  Variable* _attn_nw;
  Variable* _attn_nb;

  // shape related
  size_t _batch_dim;
  size_t _batch_heads;
  size_t _batch_tokens;
  size_t _src_seq_len;
  size_t _trg_seq_len;
  size_t _trg_batch_tokens;
  size_t _layer_id;
  size_t _max_batch_tokens;
  size_t _max_seq_len;
  size_t _hidden_size;
  size_t _heads;
  int _step;
  bool _is_pre_ln;
  bool _is_continuous_cache;

  // tensor slice
  Variable* q_out;
  Variable* k_out;
  Variable* v_out;

 public:
  DecSelfAttentionLayer(size_t layer_id, size_t max_batch_tokens,
                        size_t max_seq_len, size_t hidden_size,
                        size_t num_heads, float attn_prob_dropout_ratio,
                        float hidden_output_dropout_ratio,
                        bool is_pre_ln,
                        bool is_continuous_cache = true);

  virtual ~DecSelfAttentionLayer() {}

  std::tuple<Variable*, Variable*, Variable*> operator()(Variable* inp,
                                                         Variable* cache_k,
                                                         Variable* cache_v);

  void before_forward(size_t batch_size, size_t trg_seq_len, int steps);

  size_t load_para_and_grad(const T1* para_ptr, T2* grad_ptr);

  int load_params(const std::vector<const T1*>& para_vec, int offset);
};

template class DecSelfAttentionLayer<float, float>;
#ifdef LIGHTSEQ_cuda
template class DecSelfAttentionLayer<__half, __half>;
#endif

template <class T1, class T2>
using DecSelfAttentionLayerPtr = std::shared_ptr<DecSelfAttentionLayer<T1, T2>>;

}  // namespace lightseq
