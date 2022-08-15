#pragma once
#include "declaration.h"
#include "node.h"

namespace lightseq {

template <class T1, class T2>
class AddOperator : public Operator {
 private:
  size_t _size;
  size_t _max_size;

 public:
  AddOperator(size_t max_size) : Operator("add_op"), _max_size(max_size) {}

  Variable* operator()(Variable* inp_1, Variable* inp_2) {
    Variable* result =
        new Variable(this->_name + "-out", _max_size * sizeof(T1), _max_size * sizeof(T2));
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

    T1* temp_out = (T1*)malloc(sizeof(T1) * _max_size);
    T1* temp_inpA = (T1*)malloc(sizeof(T1) * _max_size);
    T1* temp_inpB = (T1*)malloc(sizeof(T1) * _max_size);

    cudaMemcpy(temp_inpA, inpA_ptr, sizeof(T1) * _max_size,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(temp_inpB, inpB_ptr, sizeof(T1) * _max_size,
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < this->_size; i++) {
      temp_out[i] = temp_inpA[i] + temp_inpB[i];
    }

    cudaMemcpy(out_ptr, temp_out, sizeof(T1) * _max_size,
               cudaMemcpyHostToDevice);
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
