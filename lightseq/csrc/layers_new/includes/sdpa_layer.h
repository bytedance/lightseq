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
  SDPALayer(int max_batch_tokens, int max_seq_len, int head_dim, int num_heads,
            float attn_prob_dropout_ratio);

  virtual ~SDPALayer() {}

  Variable* operator()(Variable* query, Variable* key, Variable* value,
                       Variable* mask = nullptr);

  void before_forward(int batch_size, int query_len, int kv_len, int kv_size,
                      bool mask_future);

  int load_para_and_grad(const T1* para_ptr, T2* grad_ptr);

  int load_params(const std::vector<const T1*>& para_vec, int offset);
};

template class SDPALayer<__half, __half>;
template class SDPALayer<float, float>;

template <class T1, class T2>
using SDPALayerPtr = std::shared_ptr<SDPALayer<T1, T2>>;

}  // namespace lightseq
