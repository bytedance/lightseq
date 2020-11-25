#include "util.h"

namespace lightseq {
namespace cuda {

template <typename T>
void print_vec(const thrust::device_vector<T>& outv, std::string outn,
               int num_output_ele) {
  std::cout << outn << ": ";
  if (num_output_ele > 0) {
    num_output_ele = std::min(size_t(num_output_ele), outv.size());
    thrust::copy(outv.begin(), outv.begin() + num_output_ele,
                 std::ostream_iterator<T>(std::cout, " "));
    std::cout << " ...";
  } else {
    thrust::copy(outv.begin(), outv.end(),
                 std::ostream_iterator<T>(std::cout, " "));
  }
  std::cout << std::endl;
}

template void print_vec<float>(const thrust::device_vector<float>& outv,
                               std::string outn, int num_output_ele);

template void print_vec<int>(const thrust::device_vector<int>& outv,
                             std::string outn, int num_output_ele);

template <typename T>
void print_vec(thrust::device_ptr<T> outv, std::string outn,
               int num_output_ele) {
  std::cout << outn << ": ";
  thrust::copy(outv, outv + num_output_ele,
               std::ostream_iterator<T>(std::cout, ", "));
  std::cout << std::endl;
}

template void print_vec<float>(thrust::device_ptr<float> outv, std::string outn,
                               int num_output_ele);

template void print_vec<int>(thrust::device_ptr<int> outv, std::string outn,
                             int num_output_ele);

template <typename T>
void print_vec(const T* outv, std::string outn, int num_output_ele) {
  std::cout << outn << ": ";
  std::vector<T> hout(num_output_ele, (T)0);
  cudaMemcpy(hout.data(), outv, num_output_ele * sizeof(T),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < num_output_ele; i++) {
    std::cout << hout[i] << ", ";
  }
  std::cout << std::endl;
}

template <>
void print_vec<__half>(const __half* outv, std::string outn,
                       int num_output_ele) {
  std::cout << outn << ": ";
  std::vector<__half> hout(num_output_ele, (__half)0.f);
  cudaMemcpy(hout.data(), outv, num_output_ele * sizeof(__half),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < num_output_ele; i++) {
    std::cout << __half2float(hout[i]) << ", ";
  }
  std::cout << std::endl;
}

template void print_vec<float>(const float* outv, std::string outn,
                               int num_output_ele);

template void print_vec<int>(const int* outv, std::string outn,
                             int num_output_ele);

template void print_vec<__half>(const __half* outv, std::string outn,
                                int num_output_ele);

template <typename T>
void print_vec(const T* outv, std::string outn, int start, int end) {
  std::cout << outn << ": ";
  thrust::copy(thrust::device_pointer_cast(outv + start),
               thrust::device_pointer_cast(outv + end),
               std::ostream_iterator<T>(std::cout, ", "));
  std::cout << std::endl;
}

template <>
void print_vec<__half>(const __half* outv, std::string outn, int start,
                       int end) {
  std::cout << outn << ": ";
  int num_elements = end - start;
  std::vector<__half> hout(num_elements, (__half)0.f);
  cudaMemcpy(hout.data(), outv + start, num_elements * sizeof(__half),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < num_elements; i++) {
    std::cout << __half2float(hout[i]) << ", ";
  }
  std::cout << std::endl;
}

template void print_vec<float>(const float* outv, std::string outn, int start,
                               int end);

template void print_vec<int>(const int* outv, std::string outn, int start,
                             int end);
void print_time_duration(
    const std::chrono::high_resolution_clock::time_point& start,
    std::string duration_name, cudaStream_t stream) {
  cudaStreamSynchronize(stream);
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << duration_name
            << " duration time is: " << (elapsed).count() * 1000 << " ms"
            << std::endl;
  return;
}

struct prg_uniform {
  float a, b;

  __host__ __device__ prg_uniform(float _a = 0.f, float _b = 1.f)
      : a(_a), b(_b){};

  __host__ __device__ float operator()(const unsigned int n) const {
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist(a, b);
    rng.discard(n);
    return dist(rng);
  }
};

struct prg_norm {
  float a, b;

  __host__ __device__ prg_norm(float _a = 0.f, float _b = 1.f) : a(_a), b(_b){};

  __host__ __device__ float operator()(const unsigned int n) const {
    thrust::default_random_engine rng;
    thrust::random::normal_distribution<float> dist(a, b);
    rng.discard(n);
    return dist(rng);
  }
};

void generate_distribution(thrust::device_vector<float>& input_output,
                           std::string mode, float a, float b) {
  thrust::counting_iterator<unsigned int> index_sequence_begin(0);
  if (mode == "uniform")
    thrust::transform(index_sequence_begin,
                      index_sequence_begin + input_output.size(),
                      input_output.begin(), prg_uniform(a, b));
  if (mode == "norm")
    thrust::transform(index_sequence_begin,
                      index_sequence_begin + input_output.size(),
                      input_output.begin(), prg_norm(a, b));
}

void read_batch_tokenids_from_file(std::string file_name, int& batch_size,
                                   int& batch_seq_len,
                                   std::vector<int>& input_ids) {
  std::ifstream fin(file_name);
  fin >> batch_size >> batch_seq_len;
  input_ids = std::vector<int>(batch_size * batch_seq_len, 0);
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < batch_seq_len; j++) {
      int idx = i * batch_seq_len + j;
      fin >> input_ids[idx];
    }
  }
}

}  // namespace cuda
}  // namespace lightseq
