/*
  Copyright (c) 2022 - 2023, Bytedance, The LightSeq Team
*/

#pragma once
#include <string>
#include <memory>
#include <map>
#include <iostream>
#include <vector>
#include <algorithm>

#include "declaration.h"
#include "allocator.h"

namespace lightseq {
enum LSMemoryType { FixedMemory, SharedMemory, OffsetMemory };

class TensorUsage {
 public:
  int first_idx, last_idx;
  int unique_id;
  size_t size;
  std::string _name;
  TensorUsage(int uid, int fidx, int lidx, size_t s, std::string name)
      : first_idx(fidx), last_idx(lidx), unique_id(uid), size(s), _name(name) {}
  ~TensorUsage() = default;
};

class MemoryManager {
 private:
  std::vector<char*> buffer_vec_;
  char* buffer_ = nullptr;
  std::map<int, TensorUsage> tensor_usages_;
  size_t buffer_size_;
  std::map<int, char*> tensor_ptr;
  AllocatorPtr _allocator_ptr;

 public:
  MemoryManager() : _allocator_ptr(new Allocator()) {}
  virtual ~MemoryManager() {}

  char* get_memory(int unique_id) { return tensor_ptr.find(unique_id)->second; }

  void update_tensor_life_idx(int unique_id, int node_idx, size_t size,
                              std::string name);

  void remove_life_cycle(int unique_id);

  void calculate_buffer_();

  size_t buffer_size() { return buffer_size_; }

  AllocatorPtr allocator() { return _allocator_ptr; }
};
}  // namespace lightseq
