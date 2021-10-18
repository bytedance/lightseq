#include "bert.h"

int main(int argc, char* argv[]) {
  std::string model_weights_path = argv[1];
  int max_batch_size = 128;

  auto model = lightseq::cuda::Bert(model_weights_path, max_batch_size);

  int batch_size = 1;
  std::vector<int> host_input = {101, 4931, 1010, 2129, 2024, 2017, 102, 0};
  int batch_seq_len = host_input.size();

  int* d_input;
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
  const lightseq::cuda::OperationTypeTraits<bert_optype>::DataType* res = model.get_result_ptr();
  lightseq::cuda::print_vec(res, "res token", 5);
  return 0;
}
