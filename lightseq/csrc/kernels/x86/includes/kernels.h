#pragma once
#include "cstdio"
#include "util.h"

namespace lightseq {
namespace x86 {

template <typename T>
int matrix_gemm(const T* inpA, const T* inpB, T* outC, int m, int n, int k);

}  // namespace x86
}  // namespace lightseq
