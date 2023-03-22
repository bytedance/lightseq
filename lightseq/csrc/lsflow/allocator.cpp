#include "allocator.h"

namespace lightseq {

Allocator::Allocator() { _ptr_set.clear(); }

Allocator::~Allocator() {
  auto _tmp_ptr_set = _ptr_set;
  for (auto iter : _tmp_ptr_set) {
    try {
      free_mem(iter);
    } catch (...) {
      // printf("execute ~Allocator() free_mem %p failed!\n", iter);
    }
  }
  _ptr_set.clear();
}

char* Allocator::malloc_mem(size_t size) {
  char* ptr = nullptr;

  try {
#ifdef LIGHTSEQ_cuda
    ptr = cuda::cuda_malloc<char>(size);
#else
    ptr = (char*)malloc(size);
#endif
  } catch (...) {
    std::string error_message =
        "allocate memory failed! size is: " + std::to_string((size / MB_SIZE)) +
        " MB\n";
    printf("%s", error_message.c_str());
    throw std::runtime_error(error_message);
  }
  if (_ptr_set.find(ptr) != _ptr_set.end()) {
    printf("allocate same address with twice.\n");
    throw std::runtime_error("allocate same address with twice.\n");
  }
  _ptr_set.insert(ptr);
  return ptr;
}

void Allocator::free_mem(char* ptr) {
  if (_ptr_set.find(ptr) == _ptr_set.end() || ptr == nullptr) {
    return;
  }
  _ptr_set.erase(ptr);
#ifdef LIGHTSEQ_cuda
  cuda::cuda_free(ptr);
#else
  free(ptr);
#endif
}

}  // namespace lightseq
