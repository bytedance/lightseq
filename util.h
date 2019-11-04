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

static const char *_cudaGetErrorString(cudaError_t error) {
  return cudaGetErrorString(error);
}

static const char *_cudaGetErrorString(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "CUBLAS_UNKNOW";
}

template <typename T>
void check_gpu_error(T result, char const *const func, const char *const file, int const line) {
  if (result) {
    throw std::runtime_error(std::string("[CUDA][ERROR] ") + \
        + file + "(" + std::to_string(line) + "): " + \
        (_cudaGetErrorString(result)) + "\n");
  }
}

#define CHECK_GPU_ERROR(val) lab::nmt::check_gpu_error((val), #val, __FILE__, __LINE__)

enum class OperationType{FP32, FP16};

template <OperationType OpType_> class OperationTypeTraits;

template <> class OperationTypeTraits<OperationType::FP32> {
public:
  typedef float DataType;
  static cudaDataType_t const computeType = CUDA_R_32F;
  static cudaDataType_t const AType = CUDA_R_32F;
  static cudaDataType_t const BType = CUDA_R_32F;
  static cudaDataType_t const CType = CUDA_R_32F;
  // add FP32 Traits here
};

template <> class OperationTypeTraits<OperationType::FP16> {
public:
  typedef __half DataType;
  static cudaDataType_t const computeType = CUDA_R_16F;
  static cudaDataType_t const AType = CUDA_R_16F;
  static cudaDataType_t const BType = CUDA_R_16F;
  static cudaDataType_t const CType = CUDA_R_16F;
  // add FP16 Traits here
};

template <typename T>
void print_vec(const thrust::device_vector<T>& outv, std::string outn,
                      int num_output_ele = -1);

template <typename T>
void print_vec(thrust::device_ptr<T> outv, std::string outn,
                      int num_output_ele);

template <typename T>
void print_vec(const T* outv, std::string outn, int num_output_ele);

void print_time_duration(
    const std::chrono::high_resolution_clock::time_point& start,
    std::string duration_name, cudaStream_t stream=0);

void generate_distribution(thrust::device_vector<float>& input_output,
                           std::string mode = "uniform", float a = 0.f,
                           float b = 1.f);

}  // namespace nmt
}  // namespace lab
