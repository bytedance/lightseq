#include "tensor.h"

namespace lightseq {
int Tensor::global_tensor_id = 0;
Tensor::Tensor(std::string name, size_t size)
    : _id(global_tensor_id++), _ctx_ptr(Context::global_instance().get()) {
  std::string prefix_name =
      _ctx_ptr->last_node() ? (_ctx_ptr->last_node()->name() + ":") : "";

  _name = prefix_name + name;
  _size = size;
  _mtype = size > 0 ? LSMemoryType::SharedMemory : LSMemoryType::FixedMemory;
  if (_mtype == LSMemoryType::SharedMemory) {
    _mm_ptr = _ctx_ptr->memory_manager_ptr();
    _ctx_ptr->mx_tensor_size = std::max(_ctx_ptr->mx_tensor_size, _size);
  }
}

void Tensor::set_tensor(char* inp) {
  if (_mtype == LSMemoryType::SharedMemory) {
    printf("set_tensor for %s, which is SharedMemory!\n", _name.c_str());
    exit(-1);
  }
  _ptr = inp;
}

void Tensor::set_tensor(const char* inp) { set_tensor(const_cast<char*>(inp)); }

char* Tensor::tensor(bool is_open_interval, bool just_view) {
  if (_mtype == LSMemoryType::FixedMemory) {
    if (!_ctx_ptr->is_built() && _ptr == nullptr) {
      return _ctx_ptr->temporary_buffer_;
    }
    return _ptr;
  }
  if (_ptr == nullptr) {
    if (!_ctx_ptr->is_built() && !just_view) {
      update_life_idx(_ctx_ptr->node_idx() - is_open_interval);
      return _ctx_ptr->temporary_buffer_;
    }
    _ptr = _mm_ptr->get_memory(_id);
  }
  return _ptr;
}

void Tensor::update_life_idx(int node_idx) {
  if (_mtype == LSMemoryType::FixedMemory) {
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
  if (_mtype == LSMemoryType::FixedMemory) {
    return;
  }
  this->remove_life_cycle();
  // *this = Tensor(this->_name, 0, true);
  _mtype = LSMemoryType::FixedMemory;
  _size = 0;
}

std::string Tensor::memory_type() {
  if (_mtype == LSMemoryType::FixedMemory) {
    return "FixedMemory";
  }
  else if (_mtype == LSMemoryType::SharedMemory) {
    return "SharedMemory";
  }

  return "Undefined";
}

}  // namespace lightseq
