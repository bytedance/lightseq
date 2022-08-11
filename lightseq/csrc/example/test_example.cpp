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

  int* arr = (int*)malloc(mx_size * 4 * sizeof(int));
  for (int i = 0; i < mx_size * 4; i++) {
    arr[i] = i * 100;
  }
  int* wei1 = arr;
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

  int* input_ptr = (int*)malloc(mx_size * sizeof(int));
  for (int i = 0; i < 10; i++) {
    input_ptr[i] = i;
  }
  input->set_value(input_ptr);

  int* output_ptr = (int*)malloc(size * sizeof(int));
  for (int i = 0; i < size; i++) {
    output_ptr[i] = i;
  }
  output->set_value(output_ptr);

  layer->forward();

  for (int i = 0; i < mx_size; i++) {
    printf("%d ", output_ptr[i]);
  }
  printf("\n----------Step.x----------\n");
}

}  // namespace lightseq

int main() {
  lightseq::pybind_test_2_layer();

  lightseq::pybind_test_2_layer();
}
