#pragma once

#include <math_constants.h>
#include <chrono>
#include <iostream>
#include <string>

#include <cublas_v2.h>
#include <cuda.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

namespace lab {
namespace nmt {

#define CUDA_CALL(f)                                                \
  {                                                                 \
    cudaError_t err = (f);                                          \
    if (err != cudaSuccess) {                                       \
      std::cout << "    CUDA Error occurred: " << err << std::endl; \
      std::exit(1);                                                 \
    }                                                               \
  }

#define CUBLAS_CALL(f)                                                \
  {                                                                   \
    cublasStatus_t err = (f);                                         \
    if (err != CUBLAS_STATUS_SUCCESS) {                               \
      std::cout << "    CuBLAS Error occurred: " << err << std::endl; \
      std::exit(1);                                                   \
    }                                                                 \
  }

template <typename T>
inline void print_vec(const thrust::device_vector<T>& outv, std::string outn,
                      int num_output_ele = -1) {
  std::cout << outn << ": ";
  if (num_output_ele > 0) {
    num_output_ele = min(size_t(num_output_ele), outv.size());
    thrust::copy(outv.begin(), outv.begin() + num_output_ele,
                 std::ostream_iterator<T>(std::cout, " "));
    std::cout << " ...";
  } else {
    thrust::copy(outv.begin(), outv.end(),
                 std::ostream_iterator<T>(std::cout, " "));
  }
  std::cout << std::endl;
}

template <typename T>
inline void print_vec(thrust::device_ptr<T> outv, std::string outn,
                      int num_output_ele) {
  std::cout << outn << ": ";
  thrust::copy(outv, outv + num_output_ele,
               std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl;
}

template <typename T>
inline void print_vec(const T* outv, std::string outn, int num_output_ele) {
  std::cout << outn << ": ";
  thrust::copy(thrust::device_pointer_cast(outv),
               thrust::device_pointer_cast(outv + num_output_ele),
               std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl;
}

void print_time_duration(
    const std::chrono::high_resolution_clock::time_point& start,
    std::string duration_name);

void generate_distribution(thrust::device_vector<float>& input_output,
                           std::string mode = "uniform", float a = 0.f,
                           float b = 1.f);

}  // namespace nmt
}  // namespace lab
