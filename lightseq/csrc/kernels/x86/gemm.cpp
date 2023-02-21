#include <iostream>
#include <mkl.h>

#include "kernels.h"

namespace lightseq {
namespace x86 {

// means inpA * inpB
template <>
void matrix_gemm(const float* inpA, const float* inpB, float* outC, int m,
                 int n, int k) {
  const int64_t lda = k;
  const int64_t ldb = n;
  const int64_t ldc = n;

  CBLAS_TRANSPOSE trans_a = CblasNoTrans;
  CBLAS_TRANSPOSE trans_b = CblasNoTrans;

  cblas_sgemm(CblasRowMajor, trans_a, trans_b, m, n, k, 1, inpA, lda, inpB, ldb,
              0, outC, ldc);
  return;
}

template <>
void gemm(bool a_is_packed, bool b_is_packed, bool transpose_a,
          bool transpose_b, int64_t m, int64_t n, int64_t k, float alpha,
          const uint8_t* a, int64_t lda, const int8_t* b, int64_t ldb,
          float beta, int32_t* c, int64_t ldc,
          const int32_t* a_shift_compensation) {
  const bool use_packed_api = a_is_packed || b_is_packed;

  const CBLAS_TRANSPOSE trans_a = transpose_a ? CblasTrans : CblasNoTrans;
  const CBLAS_TRANSPOSE trans_b = transpose_b ? CblasTrans : CblasNoTrans;

  // if (use_packed_api) {
  //   cblas_gemm_s8u8s32_compute(
  //       CblasRowMajor, a_is_packed ? (MKL_INT)CblasPacked : (MKL_INT)trans_a,
  //       b_is_packed ? (MKL_INT)CblasPacked : (MKL_INT)trans_b,
  //       CblasRowOffset, m, n, k, alpha, a, lda, 0, b, ldb, 0, beta, c, ldc,
  //       a_shift_compensation);
  // } else {

  cblas_gemm_s8u8s32(CblasRowMajor, trans_a, trans_b, CblasRowOffset, m, n, k,
                     alpha, a, lda, 0, b, ldb, 0, beta, c, ldc,
                     a_shift_compensation);
  // }

  return;
}

}  // namespace x86
}  // namespace lightseq
