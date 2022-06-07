#include "context.h"

Context::Context() : _stream(nullptr) {
  CHECK_GPU_ERROR(cublasCreate(&_cublasHandle));
}

Context &Context::Instance() {
  static Context _ctx;
  return _ctx;
}

void Context::set_stream(cudaStream_t stream) {
  _stream = stream;
  CHECK_GPU_ERROR(cublasSetStream(_cublasHandle, _stream));
}

cudaStream_t Context::get_stream() { return _stream; }

cublasHandle_t Context::get_cublashandle() { return _cublasHandle; }
