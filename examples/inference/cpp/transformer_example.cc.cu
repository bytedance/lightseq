#include "transformer.h"

/**
@file
Example of how to run transformer inference using our implementation.
*/

int main(int argc, char *argv[]) {
  std::string model_weights_path = argv[1];
  int max_batch_size = 128;

  auto model = lightseq::cuda::Transformer(model_weights_path, max_batch_size);

  int batch_size = 1;
  int batch_seq_len = 14;
  std::vector<int> host_input = {0,     100, 657, 14,    1816, 6, 53,
                                 50264, 473, 45,  50264, 162,  4, 2};

  int *d_input;
  lightseq::cuda::CHECK_GPU_ERROR(
      cudaMalloc(&d_input, sizeof(int) * batch_size * batch_seq_len));
  lightseq::cuda::CHECK_GPU_ERROR(cudaMemcpy(
      d_input, host_input.data(), sizeof(int) * batch_size * batch_seq_len,
      cudaMemcpyHostToDevice));
  /* ---step5. infer and log--- */
  for (int i = 0; i < 10; i++) {
    auto start = std::chrono::high_resolution_clock::now();

    model.infer(d_input, batch_size, batch_seq_len);
    lightseq::cuda::print_time_duration(start, "one infer time", 0);
  }
  return 0;
}
