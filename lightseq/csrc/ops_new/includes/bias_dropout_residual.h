#pragma once
#include "declaration.h"
#include "node.h"
#include "kernels.h"

namespace lightseq {

// transformer layer's postprocessing dropout, after attn or ffn module,
// before residual add.
template <typename T1, typename T2>
class BiasDropoutResidualOp : public Node {
 private:
  float ratio;
  bool training;

 public:
  float RATIO() const { return training ? ratio : 0.0; }

  BiasDropoutResidualOp(float r, size_t max_ele_num)
      : ratio(r), training(true) {}

  virtual ~BiasDropoutResidualOp();

  void bias_dropout_residual(T *output, const T *input, const T *residual,
                             const T *bias, int rows, int cols,
                             cudaStream_t stream);

  void d_bias_dropout_residual(T *d_input, T *d_bias, const T *d_output,
                               int rows, int cols, cudaStream_t stream);

  bool HasDropout() const;

  void SetTrainingMode(bool training);

 private:
  uint8_t *_mask;
  Config _config;
};
}  // namespace lightseq
