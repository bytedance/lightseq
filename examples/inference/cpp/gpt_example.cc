#include "gpt.h"

int main(int argc, char* argv[]) {
  std::string model_weights_path = argv[1];
  int max_batch_size = 128;

  auto model = lightseq::cuda::Gpt(model_weights_path, max_batch_size);

  int batch_size = 1;
  std::vector<int> host_input = {3666, 1438, 318, 402, 11571};
  int batch_seq_len = host_input.size();

  int* d_input;
  lightseq::cuda::CHECK_GPU_ERROR(
      cudaMalloc(&d_input, sizeof(int) * batch_size * batch_seq_len));
  lightseq::cuda::CHECK_GPU_ERROR(cudaMemcpy(
      d_input, host_input.data(), sizeof(int) * batch_size * batch_seq_len,
      cudaMemcpyHostToDevice));

  for (int i = 0; i < 10; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    model.sample(d_input, batch_size, batch_seq_len);
    lightseq::cuda::print_time_duration(start, "one infer time", 0);
  }
  const int* res = model.get_result_ptr();
  lightseq::cuda::print_vec(res, "res token", 10);

  for (int i = 0; i < 10; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    model.ppl(d_input, batch_size, batch_seq_len);
    lightseq::cuda::print_time_duration(start, "one infer time", 0);
  }
  const float* score = model.get_score_ptr();
  lightseq::cuda::print_vec(score, "res token", 1);

  return 0;
}
