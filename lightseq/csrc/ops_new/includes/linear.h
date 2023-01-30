#pragma once
#include "declaration.h"
#include "node.h"

namespace lightseq {

template <typename T1, typename T2>
class LinearOp : public Operator {
 private:
  int _output_size;
  int _input_size;
  int _max_batch_tokens;
  int _batch_tokens;
  std::array<int, 3> _gemm_algos;

#ifdef LIGHTSEQ_cuda
  cublasOperation_t _opA;
  cublasOperation_t _opB;
#endif
  float _alpha;
  MATRIX_OP _op_AA;
  MATRIX_OP _op_BB;

 public:
  LinearOp(int max_batch_tokens, int output_size, int input_size,
           MATRIX_OP opA = MATRIX_OP::Transpose,
           MATRIX_OP opB = MATRIX_OP::NonTranspose,
           //  cublasOperation_t opA = CUBLAS_OP_T,
           //  cublasOperation_t opB = CUBLAS_OP_N,
           float alpha = float(1.))
      : Operator("LinearOp"),
        _max_batch_tokens(max_batch_tokens),
        _output_size(output_size),
        _input_size(input_size),
        _op_AA(opA),
        _op_BB(opB),
#ifdef LIGHTSEQ_cuda
        _opA(opA == MATRIX_OP::Transpose ? CUBLAS_OP_T : CUBLAS_OP_N),
        _opB(opB == MATRIX_OP::Transpose ? CUBLAS_OP_T : CUBLAS_OP_N),
#endif
        _gemm_algos(std::array<int, 3>({99, 99, 99})),
        _alpha(alpha) {
  }

  ~LinearOp() {}

  Variable* operator()(Variable* inp, Variable* weight);

  void forward() override;

  void before_forward(int batch_tokens) { _batch_tokens = batch_tokens; }

  void backward() override;

  void before_backward() {}
};

}  // namespace lightseq
