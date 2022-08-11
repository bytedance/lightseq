
#pragma once
#include <string>
#include <memory>
#include <map>
#include <iostream>
#include <vector>
#include <algorithm>

#include "declaration.h"

namespace lightseq {
enum LSMemoryType { FixedMemory, SharedMemory };

enum RuntimeStatus { Inference, Training };

class TensorUsage {
 public:
  int first_idx, last_idx;
  int unique_id;
  size_t size;
  std::string _name;
  TensorUsage(int uid, int fidx, int lidx, size_t s, std::string name)
      : unique_id(uid), first_idx(fidx), last_idx(lidx), size(s), _name(name) {}
};

class MemoryManager {
 private:
  char* buffer_ = nullptr;
  std::map<int, TensorUsage> tensor_usages_;
  size_t buffer_size_;
  std::map<int, char*> tensor_ptr;

 public:
  MemoryManager() {}
  virtual ~MemoryManager() {
    if (buffer_ != nullptr) {
      delete buffer_;
    }
  }

  char* get_memory(int unique_id) { return tensor_ptr.find(unique_id)->second; }

  void update_tensor_life_idx(int unique_id, int node_idx, size_t size,
                              std::string name);

  void remove_life_cycle(int unique_id);

  void calculate_buffer_();

  size_t buffer_size() { return buffer_size_; }
};
}  // namespace lightseq
