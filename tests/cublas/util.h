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

void init_data(float *fX, __half *hX, int8_t *iX, float *fW, __half *hW,
               int8_t *iW, int C, int B, int O, int H) {
  for (int j = 0; j < C; ++j) {
    int start = j * B * H;
    for (int i = 0; i < B * H; ++i) {
      float x = float(i % 255 - 127) / 127;
      fX[start + i] = x;
      hX[start + i] = __float2half_rn(x);
      iX[start + i] = float2int8(x, 127);
    }
  }
  for (int j = 0; j < C; ++j) {
    int start = j * O * H;
    for (int i = 0; i < O * H; ++i) {
      float x = float(i % 255 - 127) / 127;
      fW[start + i] = x;
      hW[start + i] = __float2half_rn(x);
      iW[start + i] = float2int8(x, 127);
    }
  }
}

void print_res(float *Y, float *fY, __half *hY, int32_t *iY, int C, int B,
               int O, int H, bool debug) {
  float fe = 0, he = 0, ie = 0;
  printf(">>>>> compare result >>>>>\n");
  if (debug) {
    printf("oracle:\n  ");
    for (int i = 0; i < 10; ++i) printf("%.5f%c", Y[i], " \n"[i == 9]);
  }

  printf("fp32:\n  ");
  for (int i = 0; i < 10; ++i) printf("%.5f%c", fY[i], " \n"[i == 9]);
  for (int i = 0; i < C * B * O; ++i)
    fe += fabs((debug ? Y[i] : fY[i]) - fY[i]);
  printf("  diff: %.5f\n", fe / C / B / O);

  printf("fp16:\n  ");
  for (int i = 0; i < 10; ++i) printf("%.5f%c", float(hY[i]), " \n"[i == 9]);
  for (int i = 0; i < C * B * O; ++i)
    he += fabs((debug ? Y[i] : fY[i]) - float(hY[i]));
  printf("  diff: %.5f\n", he / C / B / O);

  printf("int8:\n  ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", float(iY[i]) / 127 / 127, " \n"[i == 9]);
  for (int i = 0; i < C * B * O; ++i)
    ie += fabs((debug ? Y[i] : fY[i]) - float(iY[i]) / 127 / 127);
  printf("  diff: %.5f\n", ie / C / B / O);
}
