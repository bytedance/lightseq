#pragma once
#include "declaration.h"
#include "node.h"

namespace lightseq {

template <class T1, class T2>
class AddOperator : public Operator {
 private:
  size_t _size;
  size_t _mx_size;

 public:
  AddOperator(size_t mx_size) : Operator("add_op"), _mx_size(mx_size) {}

  Variable* operator()(Variable* inp_1, Variable* inp_2) {
    Variable* result =
        new Variable(this->_name + "-out", _mx_size, sizeof(T1), sizeof(T2));
    this->set_parents({inp_1, inp_2});
    this->set_children({result});
    return result;
  }
  virtual ~AddOperator() {}

  void before_forward(size_t size) { _size = size; }

  void forward() override {
    const T1* inpA_ptr = (T1*)this->parent(0)->value();
    const T1* inpB_ptr = (T1*)this->parent(1)->value();
    T1* out_ptr = (T1*)this->child(0)->value();

    for (int i = 0; i < this->_size; i++) {
      out_ptr[i] = inpA_ptr[i] + inpB_ptr[i];
      // printf("%d %d %d\n", (int)out_ptr[i], (int)inpA_ptr[i],
      // (int)inpB_ptr[i]);
    }
  }

  void before_backward(size_t size) { _size = size; }

  void backward() override {
    this->check_override_grad();

    const T2* grad_ptr = (T2*)this->child(0)->grad();
    T2* inpA_grad_ptr = (T2*)this->parent(0)->grad();
    T2* inpB_grad_ptr = (T2*)this->parent(1)->grad();

    for (int i = 0; i < _size; i++) {
      inpA_grad_ptr[i] = grad_ptr[i];
    }

    for (int i = 0; i < _size; i++) {
      inpB_grad_ptr[i] = grad_ptr[i];
    }
  }
};

template class AddOperator<int, int>;
}  // namespace lightseq
