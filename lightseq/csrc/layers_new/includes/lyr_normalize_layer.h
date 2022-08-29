#pragma once
#include "layer_normalize.h"
#include "layer.h"

namespace lightseq {

class LyrNormalizeLayerWeight {
 public:
  LyrNormalizeLayerWeight(int hidden_size) : _hidden_size(hidden_size) {}
  char* _gamma_ptr;
  char* _betta_ptr;

  char* _grad_gamma_ptr;
  char* _grad_betta_ptr;

  int _hidden_size;

  template <class T1, class T2>
  int load_para_and_grad(const T1* para_ptr, T2* grad_ptr) {
    int offset = 0;
    _gamma_ptr = (char*)(para_ptr + offset);
    _grad_gamma_ptr = (char*)(grad_ptr + offset);
    offset += _hidden_size;

    _betta_ptr = (char*)(para_ptr + offset);
    _grad_betta_ptr = (char*)(grad_ptr + offset);
    offset += _hidden_size;

    return offset;
  }

  template <typename T>
  int load_params(const std::vector<const T*>& para_vec, int offset) {
    int size = 0;
    _gamma_ptr = (char*)para_vec[offset + size];
    size++;
    _betta_ptr = (char*)para_vec[offset + size];
    size++;

    return size;
  }
};

template int LyrNormalizeLayerWeight::load_para_and_grad(const float* para_ptr,
                                                         float* grad_ptr);
template int LyrNormalizeLayerWeight::load_para_and_grad(const __half* para_ptr,
                                                         __half* grad_ptr);

template int LyrNormalizeLayerWeight::load_params(
    const std::vector<const float*>& para_vec, int offset);
template int LyrNormalizeLayerWeight::load_params(
    const std::vector<const __half*>& para_vec, int offset);

using LyrNormalizeLayerWeightPtr = std::shared_ptr<LyrNormalizeLayerWeight>;

template <class T1, class T2>
class LyrNormalizeLayer : public Layer {
 private:
  // operators
  LayerNormalizeOp<T1, T2>* _lyr_norm_op = nullptr;

  // parameters
  Variable* _norm_gamma;
  Variable* _norm_betta;

 public:
  LyrNormalizeLayer(LyrNormalizeLayerWeightPtr norm_wt, int max_batch_tokens,
                    int hidden_size)
      : Layer("LyrNormalizeLayer"),
        _lyr_norm_op(
            new LayerNormalizeOp<T1, T2>(max_batch_tokens, hidden_size)) {
    _norm_gamma = new Variable("layer_norm_gamma", norm_wt->_gamma_ptr,
                               norm_wt->_grad_gamma_ptr);

    _norm_betta = new Variable("layer_norm_betta", norm_wt->_betta_ptr,
                               norm_wt->_grad_betta_ptr);

    this->_context_ptr->exit_layer();  // necessary
  }

  virtual ~LyrNormalizeLayer() {}

  Variable* operator()(Variable* inp) {
    this->set_inputs({inp});
    Variable* out = (*_lyr_norm_op)(inp, _norm_gamma, _norm_betta);
    this->set_outputs({out});
    return out;
  }

  void before_forward(int batch_token_num) {
    _lyr_norm_op->before_forward(batch_token_num);
  }

  void before_backward() {}
};

template class LyrNormalizeLayer<__half, __half>;
template class LyrNormalizeLayer<float, float>;

template <class T1, class T2>
using LyrNormalizeLayerPtr = std::shared_ptr<LyrNormalizeLayer<T1, T2>>;

}  // namespace lightseq
