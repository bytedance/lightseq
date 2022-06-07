#include <iostream>
#include <vector>
#include <stdexcept>

typedef std::vector<int> vi;
typedef std::pair<std::string, vi> psvi;
typedef std::vector<psvi> vpsvi;
typedef std::vector<float> vf;

inline int round_up(int v, int d) { return (v + d - 1) / d * d; }

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

template <typename T>
void print_res(float *oracle, T *res, float time, int C, int B, int O, int H,
               std::string name, bool debug) {
  float dequant_scale = 1.0;
  if (std::is_same<T, int32_t>::value) {
    dequant_scale /= (127 * 127);
  } else if (std::is_same<T, int8_t>::value) {
    dequant_scale /= (127 * 2.951 / H);
  }
  float e = 0;
  if (debug) {
    printf("oracle:\n");
    for (int i = 0; i < 10; ++i) printf("%.5f%c", oracle[i], " \n"[i == 9]);
  }

  printf("%s:\n", name.c_str());
  if (debug)
    for (int i = 0; i < 10; ++i)
      printf("%.5f%c", float(res[i]) * dequant_scale, " \n"[i == 9]);
  for (int i = 0; i < C * B * O; ++i)
    e += fabs(oracle[i] - float(res[i]) * dequant_scale);
  printf("  diff: %.3f\n", e / (C * B * O));
  printf("  time: %.3f ms\n", time);
}

void vec_pb(vpsvi &shapes, std::string name, vi shape) {
  shapes.push_back(std::make_pair(name, shape));
}
