#pragma once
#include "proto_headers.h"
#include "proto_util.h"
#include "util.h"

template <typename T>
void convert_dtype_by_gpu(float* source_addr, float* source_buffer,
                          T* target_buffer, T* target_addr, size_t size,
                          cudaStream_t stream) {
  if (std::is_same<T, __half>::value) {
    cudaMemcpyAsync(source_buffer, source_addr, size * sizeof(float),
                    cudaMemcpyDefault, stream);
    lightseq::cuda::launch_convert_dtype(source_buffer, (__half*)target_addr,
                                         size, 1024, stream);
  } else if (std::is_same<T, float>::value) {
    cudaMemcpyAsync(target_addr, source_addr, size * sizeof(float),
                    cudaMemcpyDefault, stream);
  }
}

template <typename T>
T* malloc_memory(size_t size) {
  T* buffer_addr = nullptr;
  cudaMalloc(&buffer_addr, size * sizeof(T));
  return buffer_addr;
}
