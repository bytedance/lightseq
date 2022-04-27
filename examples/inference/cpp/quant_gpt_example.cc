#include "model_base.h"
#include "gpt.h"

/**
@file
Example of how to run gpt inference using our implementation.
*/

int main(int argc, char* argv[]) {
  std::string model_weights_path = argv[1];
  int max_batch_size = 128;

  auto model = lightseq::cuda::LSModelFactory::GetInstance().CreateModel(
      "QuantGpt", model_weights_path, max_batch_size);

  int batch_size = 1;
  int batch_seq_len = 5;
  std::vector<int> host_input = {3666, 1438, 318, 402, 11571};

  void* d_input;
  lightseq::cuda::CHECK_GPU_ERROR(
      cudaMalloc(&d_input, sizeof(int) * batch_size * batch_seq_len));
  lightseq::cuda::CHECK_GPU_ERROR(cudaMemcpy(
      d_input, host_input.data(), sizeof(int) * batch_size * batch_seq_len,
      cudaMemcpyHostToDevice));

  model->set_input_ptr(0, d_input);
  model->set_input_shape(0, {batch_size, batch_seq_len});

  for (int i = 0; i < model->get_output_size(); i++) {
    void* d_output;
    std::vector<int> shape = model->get_output_max_shape(i);
    int total_size = 1;
    for (int j = 0; j < shape.size(); j++) {
      total_size *= shape[j];
    }
    lightseq::cuda::CHECK_GPU_ERROR(
        cudaMalloc(&d_output, total_size * sizeof(int)));
    model->set_output_ptr(i, d_output);
  }
  lightseq::cuda::CHECK_GPU_ERROR(cudaStreamSynchronize(0));
  std::cout << "infer preprocessing finished" << std::endl;

  /* ---step5. infer and log--- */
  for (int i = 0; i < 10; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    model->Infer();
    lightseq::cuda::print_time_duration(start, "one infer time", 0);
  }

  for (int i = 0; i < model->get_output_size(); i++) {
    const int* d_output;
    d_output = static_cast<const int*>(model->get_output_ptr(i));
    std::vector<int> shape = model->get_output_shape(i);
    std::cout << "output shape: ";
    for (int j = 0; j < shape.size(); j++) {
      std::cout << shape[j] << " ";
    }
    std::cout << std::endl;

    lightseq::cuda::print_vec(d_output, "output", 10);
  }

  return 0;
}
