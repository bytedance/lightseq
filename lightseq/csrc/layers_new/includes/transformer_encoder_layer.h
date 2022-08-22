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
class TransformerEncoderLayer : public Layer {
private:
  // operators 

  // parameters

public:
  TransformerEncoderLayer() : Layer("TransformerEncoderLayer") {
      
    this->_context_ptr->exit_layer();  // necessary
  }
  virtual ~LayerA() {}

  Variable* operator()(Variable* inp) {
      
    return output;
  }

  void before_forward(int size) {
    // op before forward
    _operator_add->before_forward(size);
    _operator_add2->before_forward(size);
  }

  void before_backward() { return; }
};

template class LayerA<int, int>;

template <class T1, class T2>
using LayerAPtr = std::shared_ptr<LayerA<T1, T2>>;


}  // namespace lightseq
