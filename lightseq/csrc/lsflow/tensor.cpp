#include "tensor.h"

namespace lightseq {
int Tensor::global_tensor_id = 0;
Tensor::Tensor(std::string name, size_t size)
    : _id(global_tensor_id++), _ctx_ptr(Context::global_instance().get()) {
  std::string prefix_name =
      _ctx_ptr->last_node() ? (_ctx_ptr->last_node()->name() + ":") : "";

  _name = prefix_name + name;
  _size = size;
  _mtype = size > 0 ? SharedMemory : FixedMemory;
  if (_mtype == SharedMemory) {
    _mm_ptr = _ctx_ptr->memory_manager_ptr();
    _ctx_ptr->mx_tensor_size = std::max(_ctx_ptr->mx_tensor_size, _size);
  }
}

void Tensor::set_tensor(char* inp) {
  if (_mtype == SharedMemory) {
    printf("set_tensor for %s, which is SharedMemory!\n", _name.c_str());
    exit(-1);
  }
  // if (!inp) {
  //   printf("set_tensor for %s with nullptr!\n", _name.c_str());
  //   exit(-1);
  // }
  _ptr = inp;
}

void Tensor::set_tensor(const char* inp) { set_tensor(const_cast<char*>(inp)); }

char* Tensor::tensor(bool is_open_interval, bool just_view) {
  if (_mtype == FixedMemory) {
    // if (!_ptr) {
    //   printf("%s is null when use, plz set first!\n", _name.c_str());
    //   exit(-1);
    // }
    if (!_ctx_ptr->built() && _ptr == nullptr) {
      return _ctx_ptr->temporary_buffer_;
    }
    return _ptr;
  }
  if (_ptr == nullptr) {
    if (!_ctx_ptr->built() && !just_view) {
      update_life_idx(_ctx_ptr->node_idx() - is_open_interval);
      return _ctx_ptr->temporary_buffer_;
    }
    _ptr = _mm_ptr->get_memory(_id);
  }
  return _ptr;
}

void Tensor::update_life_idx(int node_idx) {
  if (_mtype == FixedMemory) {
    return;
  }
  if (_size == 0) {
    return;
  }
  _mm_ptr->update_tensor_life_idx(_id, node_idx, _size, _name);
}

void Tensor::remove_life_cycle() {
  if (_mm_ptr) _mm_ptr->remove_life_cycle(_id);
}

void Tensor::reset_fixed() {
  if (_mtype == FixedMemory) {
    //_ptr = nullptr;
    return;
  }
  this->remove_life_cycle();
  *this = Tensor(this->_name, 0);
}

}  // namespace lightseq
