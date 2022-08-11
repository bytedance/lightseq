#include "tensor.h"

namespace lightseq {

Tensor::Tensor(std::string name, size_t size, bool is_shared)
    : unique_id_(global_tensor_id_++) {  // fixed memory
  tensor_size_ = size;
  _name = name;
  memory_type_ = is_shared ? SharedMemory : FixedMemory;
  context_ptr = thread_context_ptr.get();
  memory_manager_ptr = thread_context_ptr->memory_manager_ptr();
  if (is_shared) {
    thread_context_ptr->mx_tensor_size =
        std::max(thread_context_ptr->mx_tensor_size, tensor_size_);
  }
}

Tensor::Tensor(std::string name, Tensor father_tensor, size_t offset, int size)
    : unique_id_(father_tensor.unique_id()) {
  memory_type_ = father_tensor.memory_type();
  tensor_ = father_tensor.tensor() + offset;
  _name = name;
  tensor_size_ = size;
  context_ptr = thread_context_ptr.get();
  memory_manager_ptr = thread_context_ptr->memory_manager_ptr();
}

template <class T>
void Tensor::set_tensor(T* inp) {
  if (inp == nullptr) {
    return;
  }
  if (memory_type_ == SharedMemory) {
    printf("set_tensor(T* inp) Error occuried!\n");
    printf("this tensor name is: %s\n", _name.c_str());
    exit(-1);
  }
  tensor_ = (char*)inp;
}

template void Tensor::set_tensor<int>(int* inp);
template void Tensor::set_tensor<char>(char* inp);
template void Tensor::set_tensor<float>(float* inp);

char* Tensor::tensor() {
  if (tensor_ == nullptr) {
    if (!context_ptr->built()) {
      update_life_idx(context_ptr->node_idx());
      return context_ptr->temporary_buffer_;
    }
    tensor_ = memory_manager_ptr->get_memory(unique_id_);
  }
  return tensor_;
}

void Tensor::update_life_idx(int node_idx) {
  if (memory_type_ == FixedMemory) {
    return;
  }
  memory_manager_ptr->update_tensor_life_idx(unique_id_, node_idx, tensor_size_,
                                             _name);
}

void Tensor::remove_life_cycle() {
  if (memory_manager_ptr != nullptr)
    memory_manager_ptr->remove_life_cycle(unique_id_);
}

void Tensor::reset_fixed() {
  this->remove_life_cycle();
  *this = Tensor(this->_name, tensor_size_, false);
}

int Tensor::global_tensor_id_ = 0;

}  // namespace lightseq
