#pragma once
#include "bias_add_transform_20314.h"
#include "linear.h"
#include "layer.h"

namespace lightseq {

template <class T1, class T2>
class EncDecKvLayer : public Layer {
 private:
  LinearOp<T1, T2>* _kv_linear = nullptr;
  BiasAddTrans20314<T1, T2>* _bias_add_transform_20314 = nullptr;

  // parameters
  Variable* _enc_kvw;
  Variable* _enc_kvb;

  // shape related
  size_t _layer_id;
  size_t _nshared_layer;
  size_t _batch_tokens;
  size_t _max_batch_tokens;
  size_t _hidden_size;
  size_t _heads;

 public:
  EncDecKvLayer(size_t nshared_layer, size_t max_batch_tokens,
                size_t hidden_size, size_t num_heads);

  virtual ~EncDecKvLayer() {}

  Variable* operator()(Variable* enc_out);

  void before_forward(size_t batch_size, size_t seq_len);

  int load_para_and_grad(const T1* para_ptr, T2* grad_ptr);

  int load_params(const std::vector<const T1*>& para_vec, int offset);
};

template class EncDecKvLayer<float, float>;
#ifdef LIGHTSEQ_cuda
template class EncDecKvLayer<__half, __half>;
#endif

template <class T1, class T2>
using EncDecKvLayerPtr = std::shared_ptr<EncDecKvLayer<T1, T2>>;

}  // namespace lightseq
