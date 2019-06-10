#include "src/custom/transformer/util.h"

namespace lab {
namespace nmt {

void print_time_duration(
    const std::chrono::high_resolution_clock::time_point& start,
    std::string duration_name) {
  cudaDeviceSynchronize();
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

}  // namespace nmt
}  // namespace lab
