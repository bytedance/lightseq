#pragma once
#include "declaration.h"
#include "node.h"

namespace lightseq {

template <typename T1, typename T2>
class LinearOp : public Operator {
 private:
  size_t _output_size;
  size_t _input_size;
  size_t _max_batch_tokens;
  size_t _batch_tokens;
  std::array<int, 3> _gemm_algos;

  float _alpha;
  MATRIX_OP _opA;
  MATRIX_OP _opB;

  Variable* _result;

#ifdef PYBIND_INTERFACE
#define weight_op MATRIX_OP::Transpose
#else
#define weight_op MATRIX_OP::NonTranspose
#endif

 public:
  LinearOp(size_t max_batch_tokens, size_t output_size, size_t input_size,
           MATRIX_OP opA = weight_op, MATRIX_OP opB = MATRIX_OP::NonTranspose,
           float alpha = float(1.))
      : Operator("LinearOp"),
        _max_batch_tokens(max_batch_tokens),
        _output_size(output_size),
        _input_size(input_size),
        _opA(opA),
        _opB(opB),
        _gemm_algos(std::array<int, 3>({99, 99, 99})),
        _alpha(alpha) {}

  ~LinearOp() {}

  Variable* operator()(Variable* inp, Variable* weight);

  void forward() override;

  void before_forward(size_t batch_tokens) {
    _batch_tokens = batch_tokens;
    _result->set_shape({batch_tokens, _output_size});
  }

  void backward() override;

  void before_backward() {}
};

}  // namespace lightseq
