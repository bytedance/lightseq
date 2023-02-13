#include "tensor.h"

namespace lightseq {

template <typename T>
DataType g_dtype() {
  return DataType::kNotSupported;
}
#ifdef LIGHTSEQ_cuda
template <>
DataType g_dtype<__half>() {
  return DataType::kFloat16;
}
#endif
template <>
DataType g_dtype<float>() {
  return DataType::kFloat32;
}
template <>
DataType g_dtype<double>() {
  return DataType::kFloat64;
}
template <>
DataType g_dtype<int8_t>() {
  return DataType::kInt8;
}
template <>
DataType g_dtype<int16_t>() {
  return DataType::kInt16;
}
template <>
DataType g_dtype<int>() {
  return DataType::kInt32;
}
template <>
DataType g_dtype<long long>() {
  return DataType::kInt64;
}
template <>
DataType g_dtype<uint8_t>() {
  return DataType::kUInt8;
}
template <>
DataType g_dtype<uint16_t>() {
  return DataType::kUInt16;
}
template <>
DataType g_dtype<uint32_t>() {
  return DataType::kUInt32;
}
template <>
DataType g_dtype<uint64_t>() {
  return DataType::kUInt64;
}

int dtype_size(DataType dtype) {
  switch (dtype) {
    case DataType::kFloat16:
      return 2;
    case DataType::kFloat32:
      return 4;
    case DataType::kFloat64:
      return 8;
    case DataType::kInt8:
      return 1;
    case DataType::kInt16:
      return 2;
    case DataType::kInt32:
      return 4;
    case DataType::kInt64:
      return 8;
    case DataType::kByte:
      return 1;
    case DataType::kUInt8:
      return 1;
    case DataType::kUInt16:
      return 2;
    case DataType::kUInt32:
      return 4;
    case DataType::kUInt64:
      return 8;
    case DataType::kNotSupported: {
      throw std::runtime_error(
          "call dtype_size(DataType ) with kNotSupported DataType");
      return 0;
    }
  }
  throw std::runtime_error(
      "call dtype_size(DataType ) with undecalared DataType.");
  exit(-1);
}

int Tensor::global_tensor_id = 0;
Tensor::Tensor(std::string name, DataType dtype, Shape mx_shape)
    : _id(global_tensor_id++),
      _ctx_ptr(Context::global_instance().get()),
      _dtype(dtype),
      _mx_shape(mx_shape) {
  std::string prefix_name =
      _ctx_ptr->last_node() ? (_ctx_ptr->last_node()->name() + ":") : "";

  _name = prefix_name + name;
  int mx_ele_sz = _mx_shape.element_size();
  _mtype =
      mx_ele_sz > 0 ? LSMemoryType::SharedMemory : LSMemoryType::FixedMemory;
  if (_mtype == LSMemoryType::SharedMemory) {
    _mm_ptr = _ctx_ptr->memory_manager_ptr();
    _ctx_ptr->mx_tensor_size = std::max(_ctx_ptr->mx_tensor_size,
                                        (size_t)mx_ele_sz * dtype_size(_dtype));
  }
}

Tensor::Tensor(std::string name, TensorPtr ori_tensor)
    : Tensor(name, ori_tensor->dtype()) {
  _original_tensor = ori_tensor;
  _mtype = LSMemoryType::OffsetMemory;
}

void Tensor::set_tensor(char* inp) {
  if (_mtype == LSMemoryType::FixedMemory) {
    _ptr = inp;
    return;
  }
  if (_mtype == LSMemoryType::SharedMemory) {
    printf("set_tensor for %s, which is SharedMemory!\n", _name.c_str());
    return;
  }
  if (_mtype == LSMemoryType::OffsetMemory) {
    printf("set_tensor for %s, which is OffsetMemory!\n", _name.c_str());
    return;
  }
}

void Tensor::set_tensor(const char* inp) { set_tensor(const_cast<char*>(inp)); }

void Tensor::set_shape(Shape shape) { _shape = shape; }

void Tensor::set_offset(int offset, Shape shape) {
  if (_original_tensor == nullptr) {
    printf("Error! tensor %s set_offset without original tensor",
           _name.c_str());
    exit(-1);
  }
  if (_mtype != LSMemoryType::OffsetMemory) {
    printf("Error! tensor %s set_offset without original tensor",
           _name.c_str());
    exit(-1);
  }
  _shape = shape;
  _offset = offset;
}

char* Tensor::tensor(bool is_open_interval) {
  if (_mtype == LSMemoryType::OffsetMemory) {
    return _original_tensor->tensor(is_open_interval) +
           _offset * sizeof(dtype_size(_dtype));
  }
  if (_mtype == LSMemoryType::FixedMemory) {
    if (!_ctx_ptr->is_built() && _ptr == nullptr) {
      return _ctx_ptr->temporary_buffer_;
    }
    return _ptr;
  }
  if (_mtype == LSMemoryType::SharedMemory) {
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
  if (_mx_shape.element_size() == 0) {
    return;
  }
  _mm_ptr->update_tensor_life_idx(
      _id, node_idx, _mx_shape.element_size() * dtype_size(_dtype), _name);
}

void Tensor::remove_life_cycle() {
  _mtype = LSMemoryType::FixedMemory;
  if (_mm_ptr) _mm_ptr->remove_life_cycle(_id);
}

void Tensor::reset_fixed() {
  if (_mtype == LSMemoryType::FixedMemory) {
    return;
  }
  this->remove_life_cycle();
  _mtype = LSMemoryType::FixedMemory;
  _mx_shape = Shape();
}

std::string Tensor::memory_type() {
  if (_mtype == LSMemoryType::FixedMemory) {
    return "FixedMemory";
  } else if (_mtype == LSMemoryType::SharedMemory) {
    return "SharedMemory";
  } else if (_mtype == LSMemoryType::OffsetMemory) {
    return "OffsetMemory";
  }
  return "Undefined";
}

void Tensor::print_tensor(int size) {
  _ctx_ptr->synchronize();
  int ele_siz = element_size();
  size = std::min(size, ele_siz);
  switch (_dtype) {
    case DataType::kFloat16: {
#ifdef LIGHTSEQ_cuda
      print_vec((__half*)tensor(), _name, size);
      print_vec((__half*)tensor() + ele_siz - size, _name, size);
#else
      throw std::runtime_error("error! float16 can not be used without cuda!");
#endif
      break;
    }
    case DataType::kFloat32: {
      print_vec((float*)tensor(), _name, size);
      print_vec((float*)tensor() + ele_siz - size, _name, size);
      break;
    }
    case DataType::kInt32: {
      print_vec((int*)tensor(), _name, size);
      print_vec((int*)tensor() + ele_siz - size, _name, size);
      break;
    }
    case DataType::kNotSupported: {
      throw std::runtime_error(
          "error! print tensor with kNotSupported DataType");
      break;
    }
    default: {
      printf("Please add tensor printing function of %d DataType", _dtype);
    }
  }
}

}  // namespace lightseq
