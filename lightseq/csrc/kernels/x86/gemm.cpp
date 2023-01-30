#include "kernels.h"

namespace lightseq {
namespace x86 {

// means inpA * inpB
template <typename T>
int matrix_gemm(const T* inpA, const T* inpB, T* outC, int m, int n, int k) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      *(outC + i * n + j) = 0;
      for (int it = 0; it < k; it++) {
        T a = *(inpA + i * k + it);
        T b = *(inpB + it * n + j);
        *(outC + i * n + it) += a * b;
      }
    }
  }
  return 0;
}

template int matrix_gemm(const float* inpA, const float* inpB, float* outC,
                         int m, int n, int k);

}  // namespace x86
}  // namespace lightseq
