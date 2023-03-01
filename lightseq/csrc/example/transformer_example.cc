#include "model_base.h"
#include "util.h"

/**
@file
Example of how to run transformer inference using our implementation.
*/

int main(int argc, char* argv[]) {
  std::string model_weights_path = argv[1];

  std::vector<int> example_input = {63, 47,   65,  1507, 88,  74,
                                    10, 2057, 362, 9,    284, 6};
  int eg_seq_len = example_input.size();
  int batch_size = 1, batch_seq_len = example_input.size();
  if (argc == 4) {
    batch_size = atoi(argv[2]);
    batch_seq_len = atoi(argv[3]);
  }

  int max_batch_size = std::max(4, batch_size);
  std::vector<int> host_input;
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < batch_seq_len; ++j) {
      host_input.push_back(example_input[j % eg_seq_len]);
    }
  }

  auto model = lightseq::LSModelFactory::GetInstance().CreateModel(
      "Transformer", model_weights_path, max_batch_size);

  void* d_input;
  CHECK_GPU_ERROR(
      cudaMalloc(&d_input, sizeof(int) * batch_size * batch_seq_len));
  CHECK_GPU_ERROR(cudaMemcpy(
      d_input, host_input.data(), sizeof(int) * batch_size * batch_seq_len,
      cudaMemcpyHostToDevice));

//   model->benchmark_mode(true);
  model->set_input_ptr(0, d_input);
  model->set_input_shape(0, {batch_size, batch_seq_len});

  for (int i = 0; i < model->get_output_size(); i++) {
    void* d_output;
    std::vector<int> shape = model->get_output_max_shape(i);
    int total_size = 1;
    for (int j = 0; j < shape.size(); j++) {
      total_size *= shape[j];
    }
    CHECK_GPU_ERROR(
        cudaMalloc(&d_output, total_size * sizeof(int)));
    model->set_output_ptr(i, d_output);
  }
  CHECK_GPU_ERROR(cudaStreamSynchronize(0));
  std::cout << "infer preprocessing finished" << std::endl;

  std::chrono::duration<double> elapsed;
  int iter = 0;
  /* ---step5. infer and log--- */
  for (int i = 0; i < 1; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    model->Infer();
    auto finish = std::chrono::high_resolution_clock::now();
    if (i >= 5) {
      iter++;
      elapsed += finish - start;
    }
  }

  std::cout << "lightseq inference latency: " << elapsed.count() * 1000 / iter
            << " ms" << std::endl;

  for (int i = 0; i < model->get_output_size(); i++) {
    const void* d_output;
    d_output = static_cast<const float*>(model->get_output_ptr(i));
    std::vector<int> shape = model->get_output_shape(i);
    std::cout << "output shape: ";
    for (int j = 0; j < shape.size(); j++) {
      std::cout << shape[j] << " ";
    }
    std::cout << std::endl;

    if (!i)
      lightseq::print_vec((int*)d_output, "output", batch_size);
    else
      lightseq::print_vec((float*)d_output, "output", batch_size);
  }

  return 0;
}
