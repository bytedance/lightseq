#pragma once

#include <cublas_v2.h>
#include <cuda.h>

#include <iostream>
#include <string>

#include "cuda_util.h"

class Context {
 public:
  Context();
  virtual ~Context() {}
  static Context &Instance();
  void set_stream(cudaStream_t stream);
  cudaStream_t get_stream();
  cublasHandle_t get_cublashandle();

 private:
  cudaStream_t _stream;
  cublasHandle_t _cublasHandle;
};
