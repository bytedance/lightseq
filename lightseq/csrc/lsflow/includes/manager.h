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

/*
  Class: TensorUsage
  Description:
    Records the tensor's unique_id, life cycle and size information. This
    information will be recorded in the MemoryManager for memory sharing
    allocation.
*/
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

/*
  Class: MemoryManager
  Description:
    MemoryManager manages all tensor memory available for sharing. MemoryManager
    performs memory allocation planning based on the information provided by
    TensorUsage. The basic idea is to perform greedy filling. For more details,
    please refer to: https://arxiv.org/abs/2001.03288 - Algorithm.3: Greedy by
    Size for Offset Calculation

    Furthermore, considering the phenomenon of memory fragmentation, directly
    applying for a whole buffer may cause memory allocation failure. On the
    premise of ensuring that the memory of each tensor is continuous, we open up
    several small buffers to avoid the above phenomenon.
*/
class MemoryManager {
 private:
  std::vector<char*> buffer_vec_;
  std::vector<size_t> buffer_size_vec_;
  char* buffer_ = nullptr;
  std::map<int, TensorUsage> tensor_usages_;
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
