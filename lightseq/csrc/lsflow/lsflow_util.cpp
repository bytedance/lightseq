#include "lsflow_util.h"

namespace lightseq {

void print_time_duration(
    const std::chrono::high_resolution_clock::time_point &start,
    std::string duration_name) {
#ifdef LIGHTSEQ_cuda
  CHECK_GPU_ERROR(cudaStreamSynchronize(0));
#endif
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << duration_name
            << " duration time is: " << (elapsed).count() * 1000 << " ms"
            << std::endl;
  return;
}

#ifdef LIGHTSEQ_cuda
cublasOperation_t op_from_custom(MATRIX_OP op_type) {
  switch (op_type) {
    case MATRIX_OP::Transpose:
      return CUBLAS_OP_T;
    case MATRIX_OP::NonTranspose:
      return CUBLAS_OP_N;
    default: {
      std::string error_message = "undefined custom MATRIX_OP\n";
      printf("%s", error_message.c_str());
      throw std::runtime_error("undefined custom MATRIX_OP");
    }
  }
  exit(-1);
}
#endif
}  // namespace lightseq
