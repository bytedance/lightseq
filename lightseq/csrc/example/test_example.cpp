#include "iostream"
#include "vector"
#include "cstdio"
#include "algorithm"
#include "layer_example.h"
#include "context.h"

namespace lightseq {

void pybind_test_2_layer() {
  // ================= create model =================
  int mx_size = 10;
  Context::new_thread_context();
  std::vector<int> host_input;
  for (int i = 0; i < mx_size * 4; i++) {
    host_input.push_back(i * 100);
  }

  int* arr = (int*)cuda_malloc<char>(mx_size * 4 * sizeof(int));
  cudaMemcpy((void*)arr, host_input.data(), sizeof(int) * mx_size * 4,
             cudaMemcpyHostToDevice);

  //   print_vec(arr, "input_arr", mx_size * 4);

  int* wei1 = (int*)arr;
  int* wei2 = wei1 + mx_size;
  int* grad1 = wei2 + mx_size;
  int* grad2 = grad1 + mx_size;

  Layer2APtr<int, int> layer(
      new Layer2A<int, int>(mx_size, {wei1, wei2}, {grad1, grad2}));
  Variable* input = new Variable("inputA", mx_size, sizeof(int), sizeof(int));
  Variable* output = (*layer)(input);

  // ================= before forward =================
  int size = 5;
  layer->before_forward(size);

  int* input_ptr = (int*)cuda_malloc<char>(mx_size * sizeof(int));

  std::vector<int> temp_inp{};
  for (int i = 0; i < 10; i++) {
    temp_inp.push_back(i);
  }
  cudaMemcpy((void*)input_ptr, temp_inp.data(), sizeof(int) * 10,
             cudaMemcpyHostToDevice);

  //   print_vec(input_ptr, "input_ptr", mx_size);

  input->set_value(input_ptr);

  int* output_ptr = (int*)cuda_malloc<char>(mx_size * sizeof(int));
  output->set_value(output_ptr);

  layer->forward();

  print_vec(output_ptr, "output_ptr", mx_size);
  printf("----------Step.x----------\n");
}

}  // namespace lightseq

int main() {
  lightseq::pybind_test_2_layer();

  lightseq::pybind_test_2_layer();
}
