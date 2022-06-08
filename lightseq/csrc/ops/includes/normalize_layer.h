#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>

#include <fstream>
#include "cuda_util.h"
#include "kernels.h"

using namespace std;

template <typename T>
class Normalize_Layer {
 public:
  struct Config {
    uint32_t hidden_dim;
    bool use_mean;
    Config(uint32_t hidden_dim, bool use_mean = false)
        : hidden_dim(hidden_dim), use_mean(use_mean) {}
  };

  Normalize_Layer(Config config, size_t max_rows);

  ~Normalize_Layer();

  void Forward(T *ln_res, const T *inp, const T *gamma, const T *betta,
               int batch_size, cudaStream_t stream);

  /*
  residual_grad, inp_or_out, betta should be treated carefully.
  inp_or_out = input if use_mean else output
  residual_grad, betta can be nullptr.
  residual_grad will be added to dinp if it is not nullptr
    which is useful in transformer layer when pre-ln
  betta are only used to compute xhat,
    (use_mean == false) ^ (betta == nullptr) should be true
  */
  void Backward(T *gamma_grad, T *betta_grad, T *inp_grad, const T *out_grad,
                const T *residual_grad, const T *inp_or_out, const T *gamma,
                const T *betta, int batch_size, cudaStream_t stream[2]);

  bool use_mean() const;

 private:
  Config config_;
  T *vars_;
  T *means_;
};
