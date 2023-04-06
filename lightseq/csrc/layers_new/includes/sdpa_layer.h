#pragma once
#include "dropout.h"
#include "softmax.h"
#include "strided_batch_gemm.h"
#include "layer.h"

namespace lightseq {

/*
Scaled Dot Product Attention
See paper "Attention is all you need" for details.
*/
template <class T1, class T2>
class SDPALayer : public Layer {
 private:
  // operators
  StridedBatchGemmOp<T1, T2>* _attn_scores = nullptr;
  SoftmaxOp<T1, T2>* _softmax = nullptr;
  DropoutOp<T1, T2>* _attn_prob_dropout = nullptr;
  StridedBatchGemmOp<T1, T2>* _attn_context = nullptr;

  // shape related
  int _max_batch_tokens;
  int _max_seq_len;
  int _nhead;
  int _head_dim;

 public:
  SDPALayer(size_t max_batch_tokens, size_t max_seq_len, size_t head_dim,
            size_t num_heads, float attn_prob_dropout_ratio);

  virtual ~SDPALayer() {}

  // mask is for enc-self attention and enc-dec-cross attention
  Variable* operator()(Variable* query, Variable* key, Variable* value,
                       Variable* mask = nullptr);

  void before_forward(int batch_size, int query_len, int kv_len, int kv_size,
                      bool mask_future);
};

template class SDPALayer<__half, __half>;
template class SDPALayer<float, float>;

template <class T1, class T2>
using SDPALayerPtr = std::shared_ptr<SDPALayer<T1, T2>>;

}  // namespace lightseq
