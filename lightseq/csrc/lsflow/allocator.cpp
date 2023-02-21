#include "allocator.h"

namespace lightseq {

Allocator::Allocator() { _ptr_set.clear(); }

Allocator::~Allocator() {
  for (auto iter : _ptr_set) {
    Allocator::free_mem(iter);
  }
  _ptr_set.clear();
}

char* Allocator::malloc_mem(size_t size) {
  char* ptr = nullptr;
#ifdef LIGHTSEQ_cuda
  ptr = cuda::cuda_malloc<char>(size);
#else
  ptr = (char*)malloc(size);
#endif
  _ptr_set.insert(ptr);
  return ptr;
}

void Allocator::free_mem(char* ptr) {
  if (_ptr_set.find(ptr) == _ptr_set.end()) {
    return;
  }
  _ptr_set.erase(ptr);
#ifdef LIGHTSEQ_cuda
  cuda::cuda_free(ptr);
#else
  if (ptr != nullptr) {
    free(ptr);
  }
#endif
}

}  // namespace lightseq
