#include "tensor.h"

namespace lightseq {

template <typename T>
cuda::DataType g_dtype() {
  return cuda::DataType::kNotSupported;
}
#ifdef LIGHTSEQ_cuda
template <>
cuda::DataType g_dtype<__half>() {
  return cuda::DataType::kFloat16;
}
#endif
template <>
cuda::DataType g_dtype<float>() {
  return cuda::DataType::kFloat32;
}
template <>
cuda::DataType g_dtype<double>() {
  return cuda::DataType::kFloat64;
}
template <>
cuda::DataType g_dtype<int8_t>() {
  return cuda::DataType::kInt8;
}
template <>
cuda::DataType g_dtype<int16_t>() {
  return cuda::DataType::kInt16;
}
template <>
cuda::DataType g_dtype<int>() {
  return cuda::DataType::kInt32;
}
template <>
cuda::DataType g_dtype<long long>() {
  return cuda::DataType::kInt64;
}
template <>
cuda::DataType g_dtype<uint8_t>() {
  return cuda::DataType::kUInt8;
}
template <>
cuda::DataType g_dtype<uint16_t>() {
  return cuda::DataType::kUInt16;
}
template <>
cuda::DataType g_dtype<uint32_t>() {
  return cuda::DataType::kUInt32;
}
template <>
cuda::DataType g_dtype<uint64_t>() {
  return cuda::DataType::kUInt64;
}

int dtype_size(cuda::DataType dtype) {
  switch (dtype) {
    case cuda::DataType::kFloat16:
      return 2;
    case cuda::DataType::kFloat32:
      return 4;
    case cuda::DataType::kFloat64:
      return 8;
    case cuda::DataType::kInt8:
      return 1;
    case cuda::DataType::kInt16:
      return 2;
    case cuda::DataType::kInt32:
      return 4;
    case cuda::DataType::kInt64:
      return 8;
    case cuda::DataType::kByte:
      return 1;
    case cuda::DataType::kUInt8:
      return 1;
    case cuda::DataType::kUInt16:
      return 2;
    case cuda::DataType::kUInt32:
      return 4;
    case cuda::DataType::kUInt64:
      return 8;
    case cuda::DataType::kNotSupported: {
      // throw std::runtime_error(
      //     "call dtype_size(cuda::DataType ) with kNotSupported
      //     cuda::DataType");
      printf(
          "call dtype_size(cuda::DataType ) with kNotSupported "
          "cuda::DataType\n");
      return 0;
    }
  }
  throw std::runtime_error(
      "call dtype_size(cuda::DataType ) with undecalared cuda::DataType.");
  exit(-1);
}

int Tensor::global_tensor_id = 0;
Tensor::Tensor(std::string name, cuda::DataType dtype, size_t mx_shape_size)
    : _id(global_tensor_id++),
      _ctx_ptr(Context::global_instance().get()),
      _dtype(dtype),
      _mx_shape_size(mx_shape_size) {
  std::string prefix_name =
      _ctx_ptr->last_node() ? (_ctx_ptr->last_node()->name() + ":") : "";

  _name = prefix_name + name;
  _mtype = _mx_shape_size > 0 ? LSMemoryType::SharedMemory
                              : LSMemoryType::FixedMemory;
  if (_mtype == LSMemoryType::SharedMemory) {
    _mm_ptr = _ctx_ptr->memory_manager_ptr();
    if (_ctx_ptr->mx_tensor_size < _mx_shape_size * dtype_size(_dtype)) {
      _ctx_ptr->mx_tensor_size = _mx_shape_size * dtype_size(_dtype);
      _ctx_ptr->mx_tensor_name = _name;
    }
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
           _offset * dtype_size(_dtype);
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
  _mm_ptr->update_tensor_life_idx(_id, node_idx,
                                  _mx_shape_size * dtype_size(_dtype), _name);
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
  _mx_shape_size = 0;
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
  if (ele_siz == 0) {
    printf("error occurred! this tensor is %s\n", _name.c_str());
  } else {
    printf("tensor shape: ");
    for (int iter = 0; iter < shape().size(); iter++) {
      printf("%d ", shape()[iter]);
    }
    printf(", tensor dtype: %d", _dtype);

    if (_mtype == LSMemoryType::OffsetMemory) {
      printf(", offset is %d\n", _offset);
    } else {
      printf("\n");
    }
  }

  size = std::min(size, ele_siz);
  switch (_dtype) {
    case cuda::DataType::kFloat16: {
#ifdef LIGHTSEQ_cuda
      print_vec((__half*)tensor(), _name + " head", size);
      print_vec((__half*)tensor() + ele_siz - size, _name + " tail", size);
#else
      printf("error! float16 can not be used without cuda!");
      throw std::runtime_error("error! float16 can not be used without cuda!");
#endif
      break;
    }
    case cuda::DataType::kFloat32: {
      print_vec((float*)tensor(), _name + " head", size);
      print_vec((float*)tensor() + ele_siz - size, _name + " tail", size);
      break;
    }
    case cuda::DataType::kInt32: {
      print_vec((int*)tensor(), _name + " head", size);
      print_vec((int*)tensor() + ele_siz - size, _name + " tail", size);
      break;
    }
    case cuda::DataType::kNotSupported: {
      printf("error! print tensor with kNotSupported cuda::DataType");
      throw std::runtime_error(
          "error! print tensor with kNotSupported cuda::DataType");
      break;
    }
    default: {
      printf("Please add tensor printing function of %d cuda::DataType",
             _dtype);
    }
  }
}

}  // namespace lightseq
