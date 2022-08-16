#pragma once
#include "declaration.h"
#include "node.h"
#include "kernels.h"

namespace lightseq {

// dropout inside ffn.
template <typename T1, typename T2>
class BiasActDropoutOp : public Node {
 private:
  float ratio;
  bool training;

 public:
  Config(float r) : ratio(r), training(true) {}
};
float RATIO() const { return training ? ratio : 0.0; }
Dropout(float r, size_t max_ele_num) : ratio(r), training(true) {}

virtual ~Dropout();

void bias_act_dropout(T *output, const T *input, const T *bias, int rows,
                      int cols, std::string activation_fn, cudaStream_t stream);

void d_bias_act_dropout(T *d_inp_out, T *d_bias_out, const T *input,
                        const T *bias, int rows, int cols,
                        std::string activation_fn, cudaStream_t stream);

bool HasDropout() const;

void SetTrainingMode(bool training);

private:
uint8_t *_mask;
Config _config;
};
}
