#pragma once

#include <type_traits>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <functional>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace lightseq {
namespace x86 {
// The parallel_for construct is inspired by:
// https://github.com/pytorch/pytorch/blob/cc3fc786b7ba04a0918f0e817a896a09f74f7e78/aten/src/ATen/ParallelOpenMP.h

// Array smaller than this size will not be parallelized. This value could be
// smaller as the number of computations per indices increases.
constexpr int64_t GRAIN_SIZE = 32768;

template <typename T>
inline T ceil_divide(const T& x, const T& y) {
  return (x + y - 1) / y;
}

template <typename Function>
inline void parallel_for(const int64_t begin, const int64_t end,
                         const int64_t grain_size, const Function& f) {
  if (begin >= end) {
    return;
  }
#ifdef _OPENMP
  const int64_t size = end - begin;
  if (omp_get_max_threads() == 1 || omp_in_parallel() || size <= grain_size) {
    f(begin, end);
    return;
  }
#pragma omp parallel
  {
    int64_t num_threads = omp_get_num_threads();
    if (grain_size > 0) {
      num_threads = std::min(num_threads, ceil_divide(size, grain_size));
    }

    const int64_t tid = omp_get_thread_num();
    const int64_t chunk_size = ceil_divide(size, num_threads);
    const int64_t begin_tid = begin + tid * chunk_size;
    if (begin_tid < end) {
      f(begin_tid, std::min(end, chunk_size + begin_tid));
    }
  }
#else
  (void)grain_size;
  f(begin, end);
#endif
}

template <typename T1, typename T2, typename Function>
inline void unary_transform(const T1* x, T2* y, int64_t size,
                            const Function& func) {
  std::transform(x, x + size, y, func);
}

template <typename T1, typename T2, typename Function>
inline void parallel_unary_transform(const T1* x, T2* y, int64_t size,
                                     int64_t work_size, const Function& func) {
  parallel_for(0, size, GRAIN_SIZE / work_size,
               [x, y, &func](int64_t begin, int64_t end) {
                 std::transform(x + begin, x + end, y + begin, func);
               });
}

template <typename T1, typename T2, typename T3, typename Function>
inline void binary_transform(const T1* a, const T2* b, T3* c, int64_t size,
                             const Function& func) {
  std::transform(a, a + size, b, c, func);
}

template <typename T1, typename T2, typename T3, typename Function>
inline void parallel_binary_transform(const T1* a, const T2* b, T3* c,
                                      int64_t size, int64_t work_size,
                                      const Function& func) {
  parallel_for(0, size, GRAIN_SIZE / work_size,
               [a, b, c, &func](int64_t begin, int64_t end) {
                 std::transform(a + begin, a + end, b + begin, c + begin, func);
               });
}
}  // namespace x86

template <typename T>
void print_vec(const T* outv, std::string outn, int num_output_ele);

}  // namespace lightseq
