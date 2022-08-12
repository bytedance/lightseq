#pragma once
#include "layer.h"
#include "operator_example.h"

namespace lightseq {
template <class T1, class T2>
class LayerA : public Layer {
 private:
  Variable* _para = nullptr;
  AddOperator<T1, T2>* _operator_add = nullptr;

 public:
  LayerA(int mx_size, const T1* para_ptr, T2* grad_ptr) : Layer("layera") {
    _para =
        new Variable(this->_name + "-parameter", (char*)para_ptr, (char*)grad_ptr);
    _operator_add = new AddOperator<T1, T2>(mx_size);
    this->_context_ptr->exit_layer();  // necessary
  }
  virtual ~LayerA() {
    // printf("_operator_add use_count(): %d\n", _operator_add.use_count());
    // printf("~LayerA() %s\n", this->_name.c_str());
  }

  Variable* operator()(Variable* inp) {
    Variable* output = (*_operator_add)(inp, _para);
    return output;
  }

  void before_forward(int size) {
    // op before forward
    _operator_add->before_forward(size);
  }

  void before_backward() { return; }
};

template class LayerA<int, int>;

template <class T1, class T2>
using LayerAPtr = std::shared_ptr<LayerA<T1, T2>>;

template <class T1, class T2>
class Layer2A : public Layer {
 private:
  LayerAPtr<T1, T2> layer_a;
  LayerAPtr<T1, T2> layer_b;

 public:
  Layer2A(int mx_size, std::vector<const T1*> para_vec,
          std::vector<T2*> grad_vec)
      : Layer("layer2a") {
    layer_a.reset(new LayerA<T1, T2>(mx_size, para_vec[0], grad_vec[0]));
    layer_b.reset(new LayerA<T1, T2>(mx_size, para_vec[1], grad_vec[1]));
    this->_context_ptr->exit_layer();  // necessary
  }

  virtual ~Layer2A() {
    // printf("~Layer2A() %s\n", this->_name.c_str());
  }

  Variable* operator()(Variable* input) {
    Variable* lyra_out_vec = (*layer_a)(input);
    Variable* lyrb_out_vec = (*layer_b)(lyra_out_vec);
    return lyrb_out_vec;
  }

  void before_forward(int size) {
    layer_a->before_forward(size);
    layer_b->before_forward(size);
  }

  void before_forward() { return; }
};

template <class T1, class T2>
using Layer2APtr = std::shared_ptr<Layer2A<T1, T2>>;

template class Layer2A<int, int>;

}  // namespace lightseq
