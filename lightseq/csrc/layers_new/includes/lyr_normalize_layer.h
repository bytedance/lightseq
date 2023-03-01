#pragma once
#include "layer_normalize.h"
#include "layer.h"

namespace lightseq {

template <class T1, class T2>
class LyrNormalizeLayer : public Layer {
 private:
  int _hidden_size;
  int _max_batch_tokens;

  // operators
  LayerNormalizeOp<T1, T2>* _lyr_norm_op = nullptr;

  // parameters
  Variable* _norm_gamma;
  Variable* _norm_betta;

 public:
  LyrNormalizeLayer(int max_batch_tokens, int hidden_size)
      : Layer("LyrNormalizeLayer"),
        _hidden_size(hidden_size),
        _max_batch_tokens(max_batch_tokens),
        _lyr_norm_op(
            new LayerNormalizeOp<T1, T2>(max_batch_tokens, hidden_size)) {
    _norm_gamma =
        new Variable("layer_norm_gamma", g_dtype<T1>(), g_dtype<T2>());
    _norm_betta =
        new Variable("layer_norm_betta", g_dtype<T1>(), g_dtype<T2>());

    this->_context_ptr->exit_layer();  // necessary
  }

  virtual ~LyrNormalizeLayer() {}

  Variable* operator()(Variable* inp) {
    set_inputs({inp});

    Variable* out = (*_lyr_norm_op)(inp, _norm_gamma, _norm_betta);

    set_outputs({out});
    return out;
  }

  void before_forward(int batch_size, int seq_len) {
    _lyr_norm_op->before_forward(batch_size, seq_len);
  }

  void before_backward() {}

  size_t load_para_and_grad(const T1* para_ptr, T2* grad_ptr) {
    int offset = 0;

    _norm_gamma->set_value((char*)(para_ptr + offset));
    _norm_gamma->set_grad((char*)(grad_ptr + offset));
    _norm_gamma->set_shape({size_t(_hidden_size)});
    offset += _hidden_size;

    _norm_betta->set_value((char*)(para_ptr + offset));
    _norm_betta->set_grad((char*)(grad_ptr + offset));
    _norm_betta->set_shape({size_t(_hidden_size)});
    offset += _hidden_size;

    return offset;
  }

  int load_params(const std::vector<const T1*>& para_vec, int offset) {
    int size = 0;
    _norm_gamma->set_value((char*)para_vec[offset + size]), size++;
    _norm_gamma->set_shape({size_t(_hidden_size)});
    _norm_betta->set_value((char*)para_vec[offset + size]), size++;
    _norm_betta->set_shape({size_t(_hidden_size)});
    return size;
  }
};

template class LyrNormalizeLayer<float, float>;
#ifdef LIGHTSEQ_cuda
template class LyrNormalizeLayer<__half, __half>;
#endif

template <class T1, class T2>
using LyrNormalizeLayerPtr = std::shared_ptr<LyrNormalizeLayer<T1, T2>>;

}  // namespace lightseq
