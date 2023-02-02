#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include "cuda_util.h"
namespace lightseq {

template <typename T>
void print_vec(const T *outv, std::string outn, int num_output_ele) {
  std::cout << outn << " address: " << outv << std::endl;
  std::vector<T> hout(num_output_ele, (T)0);
  CHECK_GPU_ERROR(cudaStreamSynchronize(0));
  CHECK_GPU_ERROR(cudaMemcpy(hout.data(), outv, num_output_ele * sizeof(T),
                             cudaMemcpyDeviceToHost));
  printf("value: ");
  for (int i = 0; i < num_output_ele; i++) {
    std::cout << hout[i] << ", ";
  }
  std::cout << std::endl;
}

template <>
void print_vec<__half>(const __half *outv, std::string outn,
                       int num_output_ele) {
  std::cout << outn << " address: " << outv << std::endl;
  std::vector<__half> hout(num_output_ele, (__half)0.f);
  CHECK_GPU_ERROR(cudaStreamSynchronize(0));
  CHECK_GPU_ERROR(cudaMemcpy(hout.data(), outv, num_output_ele * sizeof(__half),
                             cudaMemcpyDeviceToHost));
  printf("value: ");
  for (int i = 0; i < num_output_ele; i++) {
    std::cout << __half2float(hout[i]) << ", ";
  }
  std::cout << std::endl;
}

template <>
void print_vec<int8_t>(const int8_t *outv, std::string outn,
                       int num_output_ele) {
  std::cout << outn << " address: " << outv << std::endl;
  std::vector<int8_t> hout(num_output_ele, 0);
  CHECK_GPU_ERROR(cudaStreamSynchronize(0));
  cudaMemcpy(hout.data(), outv, num_output_ele * sizeof(int8_t),
             cudaMemcpyDeviceToHost);
  printf("value: ");
  for (int i = 0; i < num_output_ele; i++) {
    std::cout << static_cast<int>(hout[i]) << ", ";
  }
  std::cout << std::endl;
}

template <>
void print_vec<uint8_t>(const uint8_t *outv, std::string outn,
                        int num_output_ele) {
  std::cout << outn << " address: " << outv << std::endl;
  std::vector<uint8_t> hout(num_output_ele, 0);
  CHECK_GPU_ERROR(cudaStreamSynchronize(0));
  cudaMemcpy(hout.data(), outv, num_output_ele * sizeof(uint8_t),
             cudaMemcpyDeviceToHost);
  printf("value: ");
  for (int i = 0; i < num_output_ele; i++) {
    std::cout << static_cast<int>(hout[i]) << ", ";
  }
  std::cout << std::endl;
}

template void print_vec<float>(const float *outv, std::string outn,
                               int num_output_ele);

template void print_vec<int>(const int *outv, std::string outn,
                             int num_output_ele);

template void print_vec<int8_t>(const int8_t *outv, std::string outn,
                                int num_output_ele);

template void print_vec<__half>(const __half *outv, std::string outn,
                                int num_output_ele);

template void print_vec<int8_t>(const int8_t *outv, std::string outn,
                                int num_output_ele);

template void print_vec<uint8_t>(const uint8_t *outv, std::string outn,
                                 int num_output_ele);

template <typename T>
void print_vec(const T *outv, std::string outn, int start, int end) {
  std::cout << outn << ": ";
  thrust::copy(thrust::device_pointer_cast(outv + start),
               thrust::device_pointer_cast(outv + end),
               std::ostream_iterator<T>(std::cout, ", "));
  std::cout << std::endl;
}

template <>
void print_vec<__half>(const __half *outv, std::string outn, int start,
                       int end) {
  std::cout << outn << ": ";
  int num_elements = end - start;
  std::vector<__half> hout(num_elements, (__half)0.f);
  CHECK_GPU_ERROR(cudaMemcpy(hout.data(), outv + start,
                             num_elements * sizeof(__half),
                             cudaMemcpyDeviceToHost));
  for (int i = 0; i < num_elements; i++) {
    std::cout << __half2float(hout[i]) << ", ";
  }
  std::cout << std::endl;
}

template void print_vec<float>(const float *outv, std::string outn, int start,
                               int end);

template void print_vec<int>(const int *outv, std::string outn, int start,
                             int end);

namespace cuda {
/* GPU function guard */
std::string _cudaGetErrorString(cudaError_t error) {
  return cudaGetErrorString(error);
}

std::string _cudaGetErrorString(cublasStatus_t error) {
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
void check_gpu_error(T result, char const *const func, const char *const file,
                     int const line) {
  if (result) {
    throw std::runtime_error(std::string("[CUDA][ERROR] ") + +file + "(" +
                             std::to_string(line) +
                             "): " + (_cudaGetErrorString(result)) + "\n");
  }
}

template void check_gpu_error<cudaError_t>(cudaError_t result,
                                           char const *const func,
                                           const char *const file,
                                           int const line);
template void check_gpu_error<cublasStatus_t>(cublasStatus_t result,
                                              char const *const func,
                                              const char *const file,
                                              int const line);

template <typename T>
T *cuda_malloc(size_t ele_num) {
  size_t byte_size = ele_num * sizeof(T);
  T *pdata = nullptr;
  CHECK_GPU_ERROR(cudaMalloc((void **)&pdata, byte_size));
  return pdata;
}

template int *cuda_malloc<int>(size_t ele_num);

template char *cuda_malloc<char>(size_t ele_num);

template float *cuda_malloc<float>(size_t ele_num);

template __half *cuda_malloc<__half>(size_t ele_num);

template uint8_t *cuda_malloc<uint8_t>(size_t ele_num);

template int8_t *cuda_malloc<int8_t>(size_t ele_num);

void cuda_free(void *pdata) {
  if (pdata != nullptr) {
    CHECK_GPU_ERROR(cudaFree(pdata));
  }
}

template <typename T>
void cuda_set(T *pdata, int value, size_t ele_num) {
  size_t byte_size = ele_num * sizeof(T);

  if (pdata != nullptr) {
    CHECK_GPU_ERROR(cudaMemset(pdata, value, byte_size));
  }
}

template void cuda_set<float>(float *pdata, int value, size_t ele_num);

template void cuda_set<__half>(__half *pdata, int value, size_t ele_num);

template void cuda_set<uint8_t>(uint8_t *pdata, int value, size_t ele_num);

template void cuda_set<int8_t>(int8_t *pdata, int value, size_t ele_num);

template <typename T>
struct _isnan {
  __device__ bool operator()(T a) const { return isnan(a); }
};

template <>
struct _isnan<__half> {
  __device__ bool operator()(const __half a) const { return __hisnan(a); }
};

template <typename T>
struct _isinf {
  __device__ bool operator()(T a) const { return isinf(a); }
};

template <>
struct _isinf<__half> {
  __device__ bool operator()(const __half a) const { return __hisinf(a); }
};

template <typename T>
void check_nan_inf(const T *data_ptr, int dsize, bool check_nan_inf,
                   std::string file, int line, cudaStream_t stream) {
  // check_nan_inf = 0 for checking nan
  // check_nan_inf = 1 for checking inf
  bool res = false;
  std::string msg = file + "(" + std::to_string(line) + "): ";
  if (check_nan_inf) {
    msg += "nan.";
    res = thrust::transform_reduce(thrust::cuda::par.on(stream), data_ptr,
                                   data_ptr + dsize, _isnan<T>(), false,
                                   thrust::logical_or<bool>());
  } else {
    msg += "inf.";
    res = thrust::transform_reduce(thrust::cuda::par.on(stream), data_ptr,
                                   data_ptr + dsize, _isinf<T>(), false,
                                   thrust::logical_or<bool>());
  }
  if (res) {
    print_vec(data_ptr, "data(head)", 20);
    print_vec(data_ptr + dsize - 20, "data(tail)", 20);
    throw std::runtime_error(msg);
  }
  std::cout << msg << " [check pass]." << std::endl;
}

template void check_nan_inf<float>(const float *data_ptr, int dsize,
                                   bool check_nan_inf, std::string file,
                                   int line, cudaStream_t stream);

template void check_nan_inf<__half>(const __half *data_ptr, int dsize,
                                    bool check_nan_inf, std::string file,
                                    int line, cudaStream_t stream);

// square<T> computes the square of a number f(x) -> x*x
template <typename T>
struct _square {
  __host__ __device__ float operator()(const T &x) const { return x * x; }
};
template <>
struct _square<__half> {
  __host__ __device__ float operator()(const __half &x) const {
    return __half2float(x) * __half2float(x);
  }
};

template <typename T>
void check_2norm(const T *data_ptr, std::string tensor_name, int dsize,
                 cudaStream_t stream) {
  // thrust::cuda::par.on(stream), data_ptr, data_ptr + dsize, _square<T>(), 0,
  float res = thrust::transform_reduce(thrust::cuda::par.on(stream), data_ptr,
                                       data_ptr + dsize, _square<T>(), 0,
                                       thrust::plus<float>());
  res = std::sqrt(res);
  std::cout << tensor_name << " norm: " << res << std::endl;
}

template void check_2norm<float>(const float *data_ptr, std::string tensor_name,
                                 int dsize, cudaStream_t stream);

template void check_2norm<__half>(const __half *data_ptr,
                                  std::string tensor_name, int dsize,
                                  cudaStream_t stream);

int getSMVersion() {
  int device{-1};
  CHECK_GPU_ERROR(cudaGetDevice(&device));
  cudaDeviceProp props;
  CHECK_GPU_ERROR(cudaGetDeviceProperties(&props, device));
  return props.major * 10 + props.minor;
}

std::string getGPUName() {
  int device{-1};
  CHECK_GPU_ERROR(cudaGetDevice(&device));
  cudaDeviceProp props;
  CHECK_GPU_ERROR(cudaGetDeviceProperties(&props, device));
  std::string full_name = std::string(props.name);
  std::vector<std::string> name_list = {"V100", "T4", "A100", "A30", "A10"};
  for (auto name : name_list) {
    if (full_name.find(name) != std::string::npos) {
      return name;
    }
  }
  return "";
}
}  // namespace cuda
}  // namespace lightseq
