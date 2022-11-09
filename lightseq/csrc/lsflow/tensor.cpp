#include "tensor.h"

namespace lightseq {
int Tensor::global_tensor_id = 0;
Tensor::Tensor(std::string name, size_t size)
    : _id(global_tensor_id ++), _ctx_ptr(Context::global_instance().get()) {
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

Tensor::Tensor(std::string name, TensorPtr ori_tensor, size_t offset)
    : Tensor(name, 0){
      _original_tensor = ori_tensor;
      _offset = offset;
      _mtype = LSMemoryType::OffsetMemory;
    }

void Tensor::set_tensor(char* inp) {
  if (_mtype == LSMemoryType::SharedMemory) {
    printf("set_tensor for %s, which is SharedMemory!\n", _name.c_str());
    exit(-1);
  }
  _ptr = inp;
}

void Tensor::set_tensor(const char* inp) { set_tensor(const_cast<char*>(inp)); }
void Tensor::set_offset(TensorPtr ori_tensor, size_t offset) {
  remove_life_cycle();
  _offset = offset;
  _original_tensor = ori_tensor;
  _mtype = LSMemoryType::OffsetMemory;
}
void Tensor::set_offset(size_t offset) {
  if(_original_tensor == nullptr) {
    printf("Error! tensor %s set_offset without original tensor", _name.c_str());
    exit(-1);
  }
  if(_mtype != LSMemoryType::OffsetMemory) {
    printf("Error! tensor %s set_offset without original tensor", _name.c_str());
    exit(-1);
  }
  _offset = offset;
}

void Tensor::remove_offset() {
  _mtype = LSMemoryType::FixedMemory;
  _ptr = nullptr;
}

char* Tensor::tensor(bool is_open_interval) {
  if (_mtype == LSMemoryType::OffsetMemory) {
    return _original_tensor->tensor(is_open_interval) + _offset;
  }
  if (_mtype == LSMemoryType::FixedMemory) {
    if (!_ctx_ptr->is_built() && _ptr == nullptr) {
      return _ctx_ptr->temporary_buffer_;
    }
    return _ptr;
  }
  if(_mtype == LSMemoryType::SharedMemory) {
    if (_ptr == nullptr) {
      if (!_ctx_ptr->is_built()) {
        update_life_idx(_ctx_ptr->node_idx() - is_open_interval);
        return _ctx_ptr->temporary_buffer_;
      }
      _ptr = _mm_ptr->get_memory(_id);
    }
    return _ptr;
  }
  printf("Error! tensor %s without _mtype!\n", _name.c_str());
  return nullptr;
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
  } else if (_mtype == LSMemoryType::SharedMemory) {
    return "SharedMemory";
  }

  return "Undefined";
}

}  // namespace lightseq
