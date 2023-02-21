/*
  Copyright (c) 2022 - 2023, Bytedance, The LightSeq Team
*/
#pragma once
#include "declaration.h"

namespace lightseq {

class Allocator {
 private:
  std::unordered_set<char*> _ptr_set;

 public:
  Allocator();
  virtual ~Allocator();
  char* malloc_mem(size_t size);
  void free_mem(char* ptr);
};

}  // namespace lightseq
