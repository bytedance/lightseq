#pragma once
#include "declaration.h"
#include "node.h"
#include "kernels.h"

namespace lightseq {

// after attention softmax
template <typename T1, typename T2>
class StridedBatchGemmOp : public Operator {
 private:
  int _m;
  int _n;
  int _k;
  size_t _max_ele_num;
  int _batch_heads;
  float _alpha;
  float _beta;
  cublasOperation_t _op_A;
  cublasOperation_t _op_B;
  std::array<int, 3> _gemm_algos;

  int _dec_layer_id;

 public:
  StridedBatchGemmOp(size_t max_ele_num, float param_alpha, float param_beta,
                     cublasOperation_t opA, cublasOperation_t opB)
      : Operator("StridedBatchGemmOp"),
        _max_ele_num(max_ele_num),
        _alpha(param_alpha),
        _beta(param_beta),
        _op_A(opA),
        _op_B(opB),
        _gemm_algos(std::array<int, 3>({99, 99, 99})) {}

  virtual ~StridedBatchGemmOp() {}

  Variable* operator()(Variable* inpA, Variable* inpB);

  void before_forward(int mm, int nn, int kk, int batch_heads) {
    _m = mm, _n = nn, _k = kk;
    _batch_heads = batch_heads;
  }

  void forward() override;

  void before_backward(int mm, int nn, int kk, int batch_heads) {
    _m = mm, _n = nn, _k = kk;
    _batch_heads = batch_heads;
  }

  void backward() override;
};
}  // namespace lightseq
