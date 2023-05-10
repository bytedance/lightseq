#include "util.h"

namespace lightseq {
namespace cuda {
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

float dequantize(unsigned char i, float scale, float clip_max) {
  return (float(i) - scale) * clip_max / scale;
}

void dequantize_array(std::vector<unsigned char>& i8, std::vector<float>& f,
                      float clip_max, float quant_range, int start, int num) {
  for (int i = start; i < start + num; ++i) {
    f[i] = dequantize(i8[i], quant_range, clip_max);
  }
}

__global__ void kernel_convert_dtype(float* source_buffer,
                                     __half* target_buffer, size_t nele) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nele) {
    return;
  }
  *(target_buffer + idx) = __float2half(*(source_buffer + idx));
}

void launch_convert_dtype(float* source_buffer, __half* target_buffer,
                          size_t size, int max_thread, cudaStream_t stream) {
  int nblock = (size + max_thread - 1) / max_thread;
  kernel_convert_dtype<<<nblock, max_thread, 0, stream>>>(source_buffer,
                                                          target_buffer, size);
}

}  // namespace cuda
}  // namespace lightseq
