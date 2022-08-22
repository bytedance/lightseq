#pragma once
#include "bias_act_dropout.h"
#include "bias_dropout_residual.h"
#include "feed_forward.h"
#include "normalize_layer.h"
#include "layer.h"

namespace lightseq {


template <class T1, class T2>
class FeedForwardLayer : public Layer {
private:
  // operators 
  NormalizeLayerOp<T1, T2>*     _ffn_ln = nullptr;
  FeedForwardOp<T1, T2>*        _ff1 = nullptr;
  BiasActDropoutOp<T1, T2>*     _ffn_activation_dropout = nullptr;
  FeedForwardOp<T1, T2>*        _ff2 = nullptr;
  BiasDropoutResOp<T1, T2>*     _ffn_dropout = nullptr;

  // parameters
  Variable* _inter_w;
  Variable* _inter_b;
  Variable* _output_w;
  Variable* _output_b;
  Variable* _ffn_nw;
  Variable* _ffn_nw;

  // shape related
  int _batch_dim;
  int _batch_heads;
  int _batch_tokens;
  
public:
  FeedForwardLayer();

  virtual ~FeedForwardLayer() {}

  Variable* operator()(Variable* inp);
  
  void before_forward(int batch_size, int seq_len);

  void before_backward() { }
};

template class FeedForwardLayer<__half, __half>;
template class FeedForwardLayer<float, float>;

template <class T1, class T2>
using FeedForwardLayer = std::shared_ptr<FeedForwardLayer<T1, T2>>;

} // namespace lightseq 