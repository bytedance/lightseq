#pragma once
#include "declaration.h"
#include "node.h"
#include "kernels.h"

namespace lightseq {
template <typename T1, typename T2>
class DropoutOp : public Node {
 private:
  float ratio;
  bool training;
  DropoutKind _dropout_kind;
  size_t _max_ele_num;

  TensorPtr _mask;

 public:
  float RATIO() const { return _context_ptr->is_training() ? ratio : 0.0; }

  Dropout(float r, size_t max_ele_num)
      : Node("Dropout"), ratio(r), _max_ele_num(max_ele_num), training(true) {
    _mask.reset(new Tensor("_mask", max_ele_num * sizeof(uint8_t)));
  }

  virtual ~Dropout();

  // after attention softmax
  void dropout(T *output, const T *input, int count, cudaStream_t stream,
               bool bwd = false);

  void d_dropout(T *d_inp_out, int count, cudaStream_t stream);

  bool HasDropout() const;

 private:
  uint8_t *_mask;
  Config _config;
};
}  // namespace lightseq
