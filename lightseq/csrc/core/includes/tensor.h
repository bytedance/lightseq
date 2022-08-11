#pragma once
#include "declaration.h"
#include "manager.h"
#include "context.h"

namespace lightseq {

class Tensor {
 private:
  LSMemoryType memory_type_;
  char* tensor_ = nullptr;
  int unique_id_ = -1;
  std::string _name;
  size_t tensor_size_;

  static int global_tensor_id_;
  MemoryManagerPtr memory_manager_ptr = nullptr;
  Context* context_ptr;

 public:
  Tensor(std::string name, size_t size, bool is_shared);

  Tensor(std::string name, Tensor father_tensor, size_t offset, int size);

  virtual ~Tensor() {}

  template <class T>
  void set_tensor(T* inp);

  template <class T>
  void set_tensor(const T* inp) {
    set_tensor(const_cast<T*>(inp));
  }

  char* tensor();

  LSMemoryType memory_type() { return memory_type_; }

  size_t size() { return tensor_size_; }
  int unique_id() { return unique_id_; }

  void update_life_idx(int node_idx);

  void remove_life_cycle();

  void reset_fixed();
};

int Tensor::global_tensor_id_ = 0;

}  // namespace lightseq
