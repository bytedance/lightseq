#pragma once
#include "declaration.h"
#include "manager.h"
#include "context.h"

namespace lightseq {

#ifdef FP16_MODE
typedef __half TENSOR_TYPE;
#else
typedef float TENSOR_TYPE;
#endif

class Tensor {
 private:
  LSMemoryType _mtype;
  char* _ptr = nullptr;
  int _id = -1;
  std::string _name;
  size_t _size;
  MemoryManagerPtr _mm_ptr = nullptr;
  Context* _ctx_ptr;

  static int global_tensor_id;
  TensorPtr _original_tensor;
  size_t _offset;

 public:
  Tensor(std::string name, size_t size);
  Tensor(std::string name, TensorPtr ori_tensor, size_t offset);

  virtual ~Tensor() {}

  void set_tensor(char* inp);
  void set_tensor(const char* inp);
  void set_offset(size_t offset);
  void set_offset(TensorPtr ori_tensor, size_t offset);
  void remove_offset();

  char* tensor(bool is_open_interval = false);

  size_t size() { return _size; }
  int unique_id() { return _id; }

  void update_life_idx(int node_idx);

  void remove_life_cycle();

  void reset_fixed();
  std::string memory_type();
  friend class Variable;

  void print_tensor(int size);
};

}  // namespace lightseq
