#pragma once

#include <string>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include "cuda_util.h"
#include "kernels.h"

template <typename T>
class Dropout {
 public:
  struct Config {
    float ratio;
    bool training;

    Config(float r) : ratio(r), training(true) {}
    float RATIO() const { return training ? ratio : 0.0; }
  };

  Dropout(const Config &config, size_t max_ele_num);

  virtual ~Dropout();

  // after attention softmax
  void dropout(T *output, const T *input, int count, cudaStream_t stream,
               bool bwd = false);

  void d_dropout(T *d_inp_out, int count, cudaStream_t stream);

  // transformer layer's postprocessing dropout, after attn or ffn module,
  // before residual add.
  void bias_dropout_residual(T *output, const T *input, const T *residual,
                             const T *bias, int rows, int cols,
                             cudaStream_t stream);

  void d_bias_dropout_residual(T *d_input, T *d_bias, const T *d_output,
                               int rows, int cols, cudaStream_t stream);

  // dropout inside ffn.
  void bias_act_dropout(T *output, const T *input, const T *bias, int rows,
                        int cols, std::string activation_fn,
                        cudaStream_t stream);

  void d_bias_act_dropout(T *d_inp_out, T *d_bias_out, const T *input,
                          const T *bias, int rows, int cols,
                          std::string activation_fn, cudaStream_t stream);

  bool HasDropout() const;

  void SetTrainingMode(bool training);

 private:
  uint8_t *_mask;
  Config _config;
};
