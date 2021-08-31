#include <stdexcept>

inline void checkCudaStatus(cudaError_t status) {
  if (status != cudaSuccess) {
    printf("cuda API failed with status %d: %s\n", status,
           cudaGetErrorString(status));
    throw std::logic_error("cuda API failed");
  }
}

inline void checkCublasStatus(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("cuBLAS API failed with status %d\n", status);
    throw std::logic_error("cuBLAS API failed");
  }
}

int8_t float2int8(float f, float scale) {
  int8_t i = int8_t(f * scale);
  if (i < -127) i = -127;
  if (i > 127) i = 127;
  return i;
}

void matmul(float *A, float *B, float *C, int bsz, int m, int n, int k) {
  for (int l = 0; l < bsz; ++l)
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) {
        int sA = l * m * k, sB = l * k * n, sC = l * m * n;
        C[sC + i * n + j] = 0;
        for (int kk = 0; kk < k; ++kk)
          C[sC + i * n + j] += A[sA + i * k + kk] * B[sB + j * k + kk];
      }
}

template <typename T, typename S>
void allocate_memory(int C, int B, int O, int H, T **X, T **W, S **Y) {
  checkCudaStatus(cudaMallocManaged(X, C * B * H * sizeof(T)));
  checkCudaStatus(cudaMallocManaged(W, C * O * H * sizeof(T)));
  checkCudaStatus(cudaMallocManaged(Y, C * B * O * sizeof(S)));
}

template <typename T, typename S>
void free_memory(T *X, T *W, S *Y) {
  checkCudaStatus(cudaFree(X));
  checkCudaStatus(cudaFree(W));
  checkCudaStatus(cudaFree(Y));
}
