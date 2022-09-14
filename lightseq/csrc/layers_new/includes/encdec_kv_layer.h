#pragma once
#include "bias_add_transform_20314.h"
#include "feed_forward.h"
#include "layer.h"

namespace lightseq {

template <class T1, class T2>
class EncDecKvLayer : public Layer {
 private:
  FeedForwardOp<T1, T2>* _kv_linear = nullptr;
  BiasAddTrans20314<T1, T2>* _bias_add_transform_20314 = nullptr;

  // parameters
  Variable* _enc_kvw;
  Variable* _enc_kvb;

  // shape related
  int _layer_id;
  int _nshared_layer;
  int _batch_tokens;
  int _max_batch_tokens;
  int _hidden_size;
  int _heads;

  static T1* _encdec_kv_buffer;
  static T2* _grad_encdec_kv_buffer;

 public:
  EncDecKvLayer(int nshared_layer, int layer_id, int max_batch_tokens, int hidden_size, int num_heads);

  virtual ~EncDecKvLayer() {}

  std::tuple<Variable*, Variable*> operator()(Variable* enc_out);

  void before_forward(int batch_size, int seq_len);

  void before_backward();
  
  int load_para_and_grad(const T1* para_ptr, T2* grad_ptr);

  int load_params(const std::vector<const T1*>& para_vec, int offset);
};

template class EncDecKvLayer<__half, __half>;
template class EncDecKvLayer<float, float>;

template <class T1, class T2>
using EncDecKvLayerPtr = std::shared_ptr<EncDecKvLayer<T1, T2>>;

}  // namespace lightseq
