#pragma once
#include "declaration.h"
#include "node.h"

namespace lightseq {

// after attention softmax
template <typename T1, typename T2>
class StridedBatchGemmOp : public Operator {
 private:
  size_t _m;
  size_t _n;
  size_t _k;
  size_t _max_ele_num;
  size_t _batch_heads;
  float _alpha;
  float _beta;
  std::array<int, 3> _gemm_algos;
  int _max_seq;
  MATRIX_OP _opA;
  MATRIX_OP _opB;

  int _dec_layer_id;

  Variable* _result;

 public:
  StridedBatchGemmOp(size_t max_ele_num, float param_alpha, float param_beta,
                     MATRIX_OP opA, MATRIX_OP opB)
      : Operator("StridedBatchGemmOp"),
        _max_ele_num(max_ele_num),
        _alpha(param_alpha),
        _beta(param_beta),
        _opA(opA),
        _opB(opB),
        _gemm_algos(std::array<int, 3>({99, 99, 99})) {}

  virtual ~StridedBatchGemmOp() {}

  Variable* operator()(Variable* inpA, Variable* inpB);

  void before_forward(size_t mm, size_t nn, size_t kk, size_t batch_heads) {
    _m = mm, _n = nn, _k = kk;
    _batch_heads = batch_heads;
    _max_seq = -1;
    // batch_heads -> [batch_size, heads]
    _result->set_shape({batch_heads, nn, mm});
  }

  void before_forward(size_t mm, size_t nn, size_t kk, size_t batch_heads,
                      int max_seq) {
    _m = mm, _n = nn, _k = kk;
    _batch_heads = batch_heads;
    _max_seq = max_seq;
    _result->set_shape({batch_heads, nn, mm});
  }

  void forward() override;

  void before_backward(int mm, int nn, int kk, int batch_heads) {
    _m = mm, _n = nn, _k = kk;
    _batch_heads = batch_heads;
  }

  void backward() override;
};
}  // namespace lightseq
