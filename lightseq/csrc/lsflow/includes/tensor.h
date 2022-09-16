#pragma once
#include "declaration.h"
#include "manager.h"
#include "context.h"

namespace lightseq {

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

 public:
  Tensor(std::string name, size_t size);

  virtual ~Tensor() {}

  void set_tensor(char* inp);

  void set_tensor(const char* inp);

  char* tensor(bool is_open_interval = false, bool just_view = false);

  size_t size() { return _size; }
  int unique_id() { return _id; }

  void update_life_idx(int node_idx);

  void remove_life_cycle();

  void reset_fixed();
};

}  // namespace lightseq
