/*
  Copyright (c) 2022 - 2023, Bytedance, The LightSeq Team
*/

#pragma once
#include "declaration.h"
#include "manager.h"
#include "context.h"
#include "shape.h"

namespace lightseq {

// Convert C++ basic data types to custom data types.
template <typename T>
DataType g_dtype();

// return byte size of DataType.
int dtype_size(DataType dtype);

class Tensor {
 private:
  LSMemoryType _mtype;
  char* _ptr = nullptr;
  DataType _dtype;

  // If mx_shape is 0, then tensor's memory type is FixedMemory or OffsetMemory.
  size_t _mx_shape_size;
  Shape _shape;

  int _id = -1;
  std::string _name;
  MemoryManagerPtr _mm_ptr = nullptr;
  Context* _ctx_ptr;

  static int global_tensor_id;
  TensorPtr _original_tensor;
  size_t _offset = 0;

 public:
  // Applies to tensors using FixedMemory and SharedMemory memory types.
  // When the mx_shape parameter is empty, it means that the tensor uses the
  // FixedMemory memory type, and then manually set the specific pointer address
  // and tensor shape.
  Tensor(std::string name, DataType dtype, size_t mx_shape_size = 0);

  // Applicable to tensors whose video memory type is OffsetMemory.
  // In this case the initialized tensor is a partial fragment of the original
  // tensor. Later, the offset value and real shape info will be set through
  // the set_offset function.
  Tensor(std::string name, TensorPtr ori_tensor);

  virtual ~Tensor() = default;

  // Set the specific memory space address and max tensor shape for the tensor
  // object. After setting, the memory space type of the tensor object is
  // changed to FixedMemroy.
  void set_tensor(char* inp);
  void set_tensor(const char* inp);

  // Just only set tensor shape for tensor object.
  void set_shape(Shape shape);

  // Set a specific offset value for a tensor whose memory type is OffsetMemory.
  // Note that the `offset` value here represents the number of elements, not
  // bytes.
  void set_offset(size_t offset, Shape shape);

  // This method executes logic differently in different situations.
  //
  // Before the context is constructed, this method will check the life cycle of
  // the SharedMemory type tensor according to the global timestamp recorded by
  // the context. When is_open_interval is true, the lifetime is updated to
  // (timestamp - 1), otherwise updated to the latest timestamp. And returns an
  // invalid pointer for subsequent possible scheduling operations
  //
  // After the context is constructed, this method will return the real pointer
  // address.
  char* tensor(bool is_open_interval = false);

  template <typename T>
  T* tensor(bool is_open_interval = false) {
    return (T*)tensor(is_open_interval);
  }

  size_t dim_t() { return _shape.view().size(); }
  int element_size() { return _shape.element_size(); }
  const size_t& mx_shape_size() const { return _mx_shape_size; }
  const std::vector<size_t>& shape() const { return _shape.view(); }
  const DataType& dtype() const { return _dtype; }

  // unique id of the tensor.
  int unique_id() { return _id; }

  // Update the lifetime of the tensor, where node_idx represents the timestamp.
  void update_life_idx(int node_idx);

  // Remove tensor life cycle.
  void remove_life_cycle();

  // Remove the life cycle information registered by the tensor from the
  // MemoryManager, do not use shared memory.
  void reset_fixed();

  // Tensor memory types are divided into three types: SharedMemory,
  // FixedMemory, OffsetMemory.
  std::string memory_type();

  // Use the corresponding data type to print the tensor according to the
  // DataType information. Print the head and tail of the tensor according
  // to the shape information. The size parameter is used to indicate the number
  // of elements to be printed separately.
  void print_tensor(int size);

  friend class Variable;
};

}  // namespace lightseq
