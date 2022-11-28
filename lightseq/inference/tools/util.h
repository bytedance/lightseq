#pragma once

#include <math_constants.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

#include <cublas_v2.h>
#include <cuda.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include "hdf5.h"

/**
@file
Util functions
*/

namespace lightseq {
namespace cuda {

/* GPU function guard */
static const char* _cudaGetErrorString(cudaError_t error) {
  return cudaGetErrorString(error);
}

static const char* _cudaGetErrorString(cublasStatus_t error) {
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
void check_gpu_error(T result, char const* const func, const char* const file,
                     int const line) {
  if (result) {
    throw std::runtime_error(std::string("[CUDA][ERROR] ") + +file + "(" +
                             std::to_string(line) +
                             "): " + (_cudaGetErrorString(result)) + "\n");
  }
}

#define CHECK_GPU_ERROR(val) check_gpu_error((val), #val, __FILE__, __LINE__)

enum class OperationType { FP32, FP16 };

/* Precision descriptor */
template <OperationType OpType_>
class OperationTypeTraits;

/* Type when fp32 */
template <>
class OperationTypeTraits<OperationType::FP32> {
 public:
  typedef float DataType;
  // cuda gemm computeType type
  static cudaDataType_t const computeType = CUDA_R_32F;
  static cudaDataType_t const AType = CUDA_R_32F;
  static cudaDataType_t const BType = CUDA_R_32F;
  static cudaDataType_t const CType = CUDA_R_32F;
};

/* Type when fp16 */
template <>
class OperationTypeTraits<OperationType::FP16> {
 public:
  typedef __half DataType;
  // cuda gemm computeType type
  static cudaDataType_t const computeType = CUDA_R_16F;
  static cudaDataType_t const AType = CUDA_R_16F;
  static cudaDataType_t const BType = CUDA_R_16F;
  static cudaDataType_t const CType = CUDA_R_16F;
};

/* Print vector stored in GPU memory, for debug */
template <typename T>
void print_vec(const thrust::device_vector<T>& outv, std::string outn,
               int num_output_ele = -1);

template <typename T>
void print_vec(thrust::device_ptr<T> outv, std::string outn,
               int num_output_ele);

template <typename T>
void print_vec(const T* outv, std::string outn, int num_output_ele);

template <typename T>
void print_vec(const T* outv, std::string outn, int start, int end);

/* Print run time, for debug */
void print_time_duration(
    const std::chrono::high_resolution_clock::time_point& start,
    std::string duration_name, cudaStream_t stream = 0);

/* Generate distribution */
void generate_distribution(thrust::device_vector<float>& input_output,
                           std::string mode = "uniform", float a = 0.f,
                           float b = 1.f);

/*
Read input token ids from file.
the first line of input file should
be two integers: batch_size and batch_seq_len.
followed by batch_size lines of
batch_seq_len integers, e.g.
2 3
666 666 666
666 666 666
*/
void read_batch_tokenids_from_file(std::string, int& batch_size,
                                   int& batch_seq_len,
                                   std::vector<int>& input_ids);

/*
Utility function for initializing
*/
bool endswith(std::string const& full, std::string const& end);

/*
Helper function of HDF5.

Return the 1D size of given hdf5 dataset if dataset is already open.
*/
int get_hdf5_dataset_size(hid_t dataset);

/*
Helper function of HDF5.

Return the 1D size of given hdf5 dataset in the given file.
*/
int get_hdf5_dataset_size(hid_t hdf5_file, std::string dataset_name);

/*
Helper function of HDF5.

Read the data of specified type `output_type` into `output_buf`.
return: the size of output data.
*/
int read_hdf5_dataset_data(
    hid_t hdf5_file, std::string dataset_name, hid_t output_type,
    void* output_buf,
    std::function<bool(int)> size_predicate = [](int x) -> bool {
      return (x < 0);
    },
    std::string extra_msg = "");

/*
Helper function of HDF5.

Read the data of specified type `output_type` into a vector<T>,
and the vector will be returned.
*/
// TODO: merge these two _float _int function together to improve readability
std::vector<float> read_hdf5_dataset_data_float(
    hid_t hdf5_file, std::string dataset_name, hid_t output_type,
    std::function<bool(int)> size_predicate = [](int x) -> bool {
      return (x < 0);
    },
    std::string extra_msg = "");

std::vector<int> read_hdf5_dataset_data_int(
    hid_t hdf5_file, std::string dataset_name, hid_t output_type,
    std::function<bool(int)> size_predicate = [](int x) -> bool {
      return (x < 0);
    },
    std::string extra_msg = "");

/*
Helper function of HDF5.

Read a scalar of specified type `output_type` into `output_buf`.

return: the size of output data.
*/
int read_hdf5_dataset_scalar(hid_t hdf5_file, std::string dataset_name,
                             hid_t output_type, void* output_buf);

class HDF5DatasetNotFoundError : public std::runtime_error {
 public:
  HDF5DatasetNotFoundError(const char* what) : runtime_error(what) {}
};

}  // namespace cuda
}  // namespace lightseq
