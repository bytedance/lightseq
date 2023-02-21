#pragma once
#include "cstdio"
#include "util.h"

namespace lightseq {
namespace x86 {

template <typename InType, typename OutType>
void matrix_gemm(const InType* inpA, const InType* inpB, OutType* outC, int m,
                 int n, int k);

template <typename AType, typename BType, typename CType>
void gemm(bool a_is_packed, bool b_is_packed, bool transpose_a,
          bool transpose_b, int64_t m, int64_t n, int64_t k, float alpha,
          const AType* a, int64_t lda, const BType* b, int64_t ldb, float beta,
          CType* c, int64_t ldc, const CType* a_shift_compensation = nullptr);

}  // namespace x86
}  // namespace lightseq
