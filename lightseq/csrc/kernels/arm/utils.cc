#include "utils.h"

namespace lightseq {

template <typename T>
void print_vec(const T *outv, std::string outn, int num_output_ele) {
  std::cout << outn << " address: " << outv << std::endl;
  printf("value: ");
  for (int i = 0; i < num_output_ele; i++) {
    std::cout << outv[i] << ", ";
  }
  std::cout << std::endl;
}

template <>
void print_vec<int8_t>(const int8_t *outv, std::string outn,
                       int num_output_ele) {
  std::cout << outn << " address: " << outv << std::endl;
  printf("value: ");
  for (int i = 0; i < num_output_ele; i++) {
    std::cout << static_cast<int>(outv[i]) << ", ";
  }
  std::cout << std::endl;
}

template <>
void print_vec<uint8_t>(const uint8_t *outv, std::string outn,
                        int num_output_ele) {
  std::cout << outn << " address: " << outv << std::endl;
  printf("value: ");
  for (int i = 0; i < num_output_ele; i++) {
    std::cout << static_cast<int>(outv[i]) << ", ";
  }
  std::cout << std::endl;
}

template void print_vec<float>(const float *outv, std::string outn,
                               int num_output_ele);

template void print_vec<int>(const int *outv, std::string outn,
                             int num_output_ele);

template void print_vec<int8_t>(const int8_t *outv, std::string outn,
                                int num_output_ele);

template void print_vec<int8_t>(const int8_t *outv, std::string outn,
                                int num_output_ele);

template void print_vec<uint8_t>(const uint8_t *outv, std::string outn,
                                 int num_output_ele);
}  // namespace lightseq
