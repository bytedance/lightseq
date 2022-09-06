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

#include "cuda_util.h"
#include "hdf5.h"

/**
@file
Util functions
*/

namespace lightseq {

/* GPU function guard */
static std::string _cudaGetErrorString(cudaError_t error) {
  return std::string(cudaGetErrorName(error)) +
         std::string(cudaGetErrorString(error));
}

static std::string _cudaGetErrorString(cublasStatus_t error) {
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

template <typename T>
T* to_gpu(const T* host_pointer, int size, cudaStream_t stream) {
  T* gpu_pointer;
  CHECK_GPU_ERROR(cudaMalloc(&gpu_pointer, size * sizeof(T)));
  CHECK_GPU_ERROR(cudaMemcpyAsync(gpu_pointer, host_pointer, size * sizeof(T),
                                  cudaMemcpyHostToDevice, stream));
  return gpu_pointer;
}

float dequantize(unsigned char i, float scale, float clip_max);

void dequantize_array(std::vector<unsigned char>& i8, std::vector<float>& f,
                      float clip_max, float quant_range, int start, int num);

void transform_param_shape(float* origin, float* buffer, int row_size,
                           int col_size);
}  // namespace lightseq
