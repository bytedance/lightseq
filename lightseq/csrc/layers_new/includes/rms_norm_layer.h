#pragma once
#include "rms_layer_norm.h"
#include "layer.h"

namespace lightseq {

template <class T1, class T2>
class RMSNormLayer : public Layer {
 private:
  int _hidden_size;
  int _max_batch_tokens;

  // operators
  RMSLayerNormalizeOp<T1, T2>* _rms_norm = nullptr;

  // parameters
  Variable* _norm_scale;

 public:
  RMSNormLayer(int max_batch_tokens, int hidden_size)
      : Layer("RMSNormLayer"),
        _hidden_size(hidden_size),
        _max_batch_tokens(max_batch_tokens),
        _rms_norm(
            new RMSLayerNormalizeOp<T1, T2>(max_batch_tokens, hidden_size, false)) {
    _norm_scale =
        new Variable("_norm_scale", g_dtype<T1>(), g_dtype<T2>());

    this->_context_ptr->exit_layer();  // necessary
  }

  virtual ~RMSNormLayer() {}

  Variable* operator()(Variable* inp) {
    set_inputs({inp});

    Variable* out = std::get<0>((*_rms_norm)(inp, _norm_scale));

    set_outputs({out});
    return out;
  }

  void before_forward(int batch_size, int seq_len) {
    _rms_norm->before_forward(batch_size, seq_len);
  }

  void before_backward() {}

  int load_params(const std::vector<const T1*>& para_vec, int offset) {
    int size = 0;
    _norm_scale->set_value((char*)para_vec[offset + size]), size++;
    _norm_scale->set_shape({size_t(_hidden_size)});
    return size;
  }
};

template class RMSNormLayer<float, float>;
#ifdef LIGHTSEQ_cuda
template class RMSNormLayer<__half, __half>;
#endif

template <class T1, class T2>
using RMSNormLayerPtr = std::shared_ptr<RMSNormLayer<T1, T2>>;

}  // namespace lightseq
