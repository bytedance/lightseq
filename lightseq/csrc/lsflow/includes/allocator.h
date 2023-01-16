#pragma once

#include "declaration.h"

namespace lightseq {

class Allocator {
 private:
  std::unordered_set<char*> _ptr_set;

 public:
  Allocator();
  ~Allocator();
  char* malloc(size_t size);
  void free(char* ptr);
};

}  // namespace lightseq
