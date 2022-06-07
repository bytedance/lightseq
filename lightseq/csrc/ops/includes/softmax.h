#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>

#include <fstream>

#include "kernels.h"

using namespace std;

template <typename T>
class Softmax {
 public:
  struct Config {
    size_t nhead;
    bool mask_future;
    Config(size_t nhead, bool mask_future = false)
        : nhead(nhead), mask_future(mask_future) {}
  };

  Softmax(Config config) : config_(config) {}

  ~Softmax() {}

  void Forward(T *vals, const T *attn_mask, int batch_size, int from_len,
               int to_len, cudaStream_t &stream, bool mask_future = false);

  void Backward(T *out_grad, const T *soft_out, int batch_size, int from_len,
                int to_len, cudaStream_t stream);

 private:
  Config config_;
};
