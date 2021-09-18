#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <cublasLt.h>

#include <iostream>
#include <string>

#include "cuda_util.h"

class Context {
 public:
  Context() : _stream(nullptr) {
    CHECK_GPU_ERROR(cublasCreate(&_cublasHandle));
    CHECK_GPU_ERROR(cublasLtCreate(&_cublasLtHandle));
  }

  virtual ~Context() {}

  static Context &Instance() {
    static Context _ctx;
    return _ctx;
  }

  void set_stream(cudaStream_t stream) {
    _stream = stream;
    CHECK_GPU_ERROR(cublasSetStream(_cublasHandle, _stream));
  }

  cudaStream_t get_stream() { return _stream; }

  cublasHandle_t get_cublashandle() { return _cublasHandle; }

  cublasLtHandle_t get_cublaslthandle() { return _cublasLtHandle; }

 private:
  cudaStream_t _stream;
  cublasHandle_t _cublasHandle;
  cublasLtHandle_t _cublasLtHandle;
};
