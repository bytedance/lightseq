#include "model_base.h"
#include "util.h"

/**
@file
Example of how to run QuantVit inference using our implementation.
*/

int main(int argc, char* argv[]) {
  std::string model_weights_path = argv[1];
  int max_batch_size = 128;
  int batch_size = 1;
  int patch_size = 16;
  int image_size = 224;

  if (argc == 5) {
    batch_size = atoi(argv[2]);
    patch_size = atoi(argv[3]);
    image_size = atoi(argv[4]);
  }
  if (batch_size > max_batch_size) {
    throw std::runtime_error("batch_size exceeds the maximum (128)!");
  }

  int total_ele = batch_size * patch_size * image_size * image_size;

  std::vector<int> host_input;
  for (int i = 0; i < total_ele; ++i) {
    host_input.push_back(i * 2.0 / total_ele - 1);
  }

  auto model = lightseq::cuda::LSModelFactory::GetInstance().CreateModel(
      "QuantVit", model_weights_path, max_batch_size);

  void* d_input;
  lightseq::cuda::CHECK_GPU_ERROR(
      cudaMalloc(&d_input, sizeof(int) * total_ele));
  lightseq::cuda::CHECK_GPU_ERROR(cudaMemcpy(d_input, host_input.data(),
                                             sizeof(int) * total_ele,
                                             cudaMemcpyHostToDevice));

  model->set_input_ptr(0, d_input);
  model->set_input_shape(0, {batch_size, patch_size, image_size, image_size});

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
    const float* d_output;
    d_output = static_cast<const float*>(model->get_output_ptr(i));
    std::vector<int> shape = model->get_output_shape(i);
    std::cout << "output shape: ";
    for (int j = 0; j < shape.size(); j++) {
      std::cout << shape[j] << " ";
    }
    std::cout << std::endl;

    lightseq::cuda::print_vec(d_output, "output", 5);
  }

  return 0;
}
