#include <cublasLt.h>
#include <cuda_runtime.h>
#include <vector>
#include "util.h"

template <typename T, typename S>
int cublas_lt_matmul(cublasLtHandle_t handle, cublasLtMatmulDesc_t matmulDesc,
                     cublasLtMatrixLayout_t XDesc, cublasLtMatrixLayout_t WDesc,
                     cublasLtMatrixLayout_t YDesc, T *A, T *B, S *C, S *alpha,
                     S *beta) {
  cublasStatus_t status;
  status = cublasLtMatmul(handle, matmulDesc, alpha, A, XDesc, B, WDesc, beta,
                          C, YDesc, C, YDesc, nullptr, nullptr, 0, 0);

  if (status == CUBLAS_STATUS_SUCCESS)
    return 1;
  else {
    return -1;
  }
}

template <typename T, typename S>
void test_lt_matmul(cublasLtHandle_t handle, int C, int B, int O, int H, T *X,
                    T *W, S *Y, S *alpha, S *beta, int iteration) {
  cudaDataType_t XType, WType, YType;
#if CUBLAS_VER_MAJOR == 11
  cublasComputeType_t ComputeType;
  cudaDataType_t scaleType;
#else
  cudaDataType_t ComputeType;
#endif
  if (std::is_same<T, float>::value) {
    XType = WType = YType = CUDA_R_32F;
#if CUBLAS_VER_MAJOR == 11
    ComputeType = CUBLAS_COMPUTE_32F;
    scaleType = CUDA_R_32F;
#else
    ComputeType = CUDA_R_32F;
#endif
  } else if (std::is_same<T, __half>::value) {
    XType = WType = YType = CUDA_R_16F;
#if CUBLAS_VER_MAJOR == 11
    ComputeType = CUBLAS_COMPUTE_16F;
    scaleType = CUDA_R_16F;
#else
    ComputeType = CUDA_R_16F;
#endif
  } else {
    printf("Not supported data type.");
    return;
  }

  int64_t strideX = B * H, strideW = O * H, strideY = B * O;
  cublasOperation_t opTrans = CUBLAS_OP_T;

  cublasLtMatrixLayout_t XDesc, WDesc, YDesc;
  checkCublasStatus(cublasLtMatrixLayoutCreate(&XDesc, XType, H, B, H));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&WDesc, WType, H, O, H));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&YDesc, YType, O, B, O));
  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
      XDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &C, sizeof(C)));
  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
      XDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideX,
      sizeof(strideX)));
  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
      WDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &C, sizeof(C)));
  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
      WDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideW,
      sizeof(strideW)));
  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
      YDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &C, sizeof(C)));
  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
      YDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideY,
      sizeof(strideY)));

  T *Wtransform;
  checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&Wtransform),
                             sizeof(T) * C * O * H));

  cublasLtMatrixLayout_t WtransformDesc;
  checkCublasStatus(
      cublasLtMatrixLayoutCreate(&WtransformDesc, WType, O, H, O));
  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
      WtransformDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &C, sizeof(C)));
  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
      WtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideW,
      sizeof(strideW)));

  cublasLtMatrixTransformDesc_t transformDesc;
  checkCublasStatus(
      cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_32F));
  checkCublasStatus(cublasLtMatrixTransformDescSetAttribute(
      transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTrans,
      sizeof(opTrans)));

  float transformAlpha = 1.0f, transformBeta = 0.0f;
  checkCublasStatus(cublasLtMatrixTransform(
      handle, transformDesc, &transformAlpha, W, WDesc, &transformBeta, NULL,
      NULL, Wtransform, WtransformDesc, 0));

  cublasLtMatmulDesc_t matmulDesc;
#if CUBLAS_VER_MAJOR == 11
  checkCublasStatus(
      cublasLtMatmulDescCreate(&matmulDesc, ComputeType, scaleType));
#else
  checkCublasStatus(cublasLtMatmulDescCreate(&matmulDesc, ComputeType));
#endif

  float total_time = 0;
  for (int i = 0; i < iteration; ++i) {
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    int success = cublas_lt_matmul(handle, matmulDesc, WtransformDesc, XDesc,
                                   YDesc, Wtransform, X, Y, alpha, beta);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (success > 0 && i >= 10) total_time += time;
  }
  if (total_time > 0) printf("%.3f ms\n", total_time / (iteration - 10));

  checkCublasStatus(cublasLtMatrixLayoutDestroy(WtransformDesc));
  checkCublasStatus(cublasLtMatrixLayoutDestroy(XDesc));
  checkCublasStatus(cublasLtMatrixLayoutDestroy(WDesc));
  checkCublasStatus(cublasLtMatrixLayoutDestroy(YDesc));
  checkCublasStatus(cublasLtMatmulDescDestroy(matmulDesc));
  checkCublasStatus(cublasLtMatrixTransformDescDestroy(transformDesc));
  cudaDeviceSynchronize();
  checkCudaStatus(cudaFree(Wtransform));
}

void test_lt_matmul_int8(cublasLtHandle_t handle, int C, int B, int O, int H,
                         int8_t *X, int8_t *W, int32_t *Y, int32_t *alpha,
                         int32_t *beta, int iteration) {
#if CUBLAS_VER_MAJOR == 11
  cublasComputeType_t ComputeType = CUBLAS_COMPUTE_32I;
  cudaDataType_t scaleType = CUDA_R_32I;
#else
  cudaDataType_t ComputeType = CUDA_R_32I;
#endif
  cudaDataType_t XType, WType, YType;
  XType = WType = CUDA_R_8I;
  YType = CUDA_R_32I;

  int64_t strideX = B * H, strideW = O * H, strideY = B * O;
  cublasOperation_t opTrans = CUBLAS_OP_T;
  cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;
  cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;

  cublasLtMatrixLayout_t XDesc, WDesc, YDesc;
  checkCublasStatus(cublasLtMatrixLayoutCreate(&XDesc, XType, H, B, H));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&WDesc, WType, H, O, H));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&YDesc, YType, O, B, O));
  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
      XDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &C, sizeof(C)));
  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
      XDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideX,
      sizeof(strideX)));
  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
      WDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &C, sizeof(C)));
  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
      WDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideW,
      sizeof(strideW)));
  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
      YDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &C, sizeof(C)));
  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
      YDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideY,
      sizeof(strideY)));

  int8_t *Xtransform, *Wtransform;
  int32_t *Ytransform;
  checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&Xtransform),
                             sizeof(int8_t) * C * B * H));
  checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&Wtransform),
                             sizeof(int8_t) * C * O * H));
  checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&Ytransform),
                             sizeof(int32_t) * C * B * O));

  int ldXtransform = 32 * B;
  int ldWtransform = 32 * O;
  int ldYtransform = 32 * B;
  cublasLtMatrixLayout_t XtransformDesc, WtransformDesc, YtransformDesc;
  checkCublasStatus(cublasLtMatrixLayoutCreate(&XtransformDesc, CUDA_R_8I, B, H,
                                               ldXtransform));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&WtransformDesc, CUDA_R_8I, O, H,
                                               ldWtransform));
  checkCublasStatus(cublasLtMatrixLayoutCreate(&YtransformDesc, CUDA_R_32I, B,
                                               O, ldYtransform));
  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
      YtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32,
      sizeof(order_COL32)));
  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
      WtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL4_4R2_8C,
      sizeof(order_COL4_4R2_8C)));
  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
      XtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32,
      sizeof(order_COL32)));
  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
      XtransformDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &C, sizeof(C)));
  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
      XtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideX,
      sizeof(strideX)));
  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
      WtransformDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &C, sizeof(C)));
  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
      WtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideW,
      sizeof(strideW)));
  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
      YtransformDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &C, sizeof(C)));
  checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
      YtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideY,
      sizeof(strideY)));

  cublasLtMatrixTransformDesc_t transformDesc;
  checkCublasStatus(
      cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_32F));
  checkCublasStatus(cublasLtMatrixTransformDescSetAttribute(
      transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTrans,
      sizeof(opTrans)));

  float transformAlpha = 1.0f, transformBeta = 0.0f;
  checkCublasStatus(cublasLtMatrixTransform(
      handle, transformDesc, &transformAlpha, X, XDesc, &transformBeta, NULL,
      NULL, Xtransform, XtransformDesc, 0));
  checkCublasStatus(cublasLtMatrixTransform(
      handle, transformDesc, &transformAlpha, W, WDesc, &transformBeta, NULL,
      NULL, Wtransform, WtransformDesc, 0));

  cublasLtMatmulDesc_t matmulDesc;
#if CUBLAS_VER_MAJOR == 11
  checkCublasStatus(
      cublasLtMatmulDescCreate(&matmulDesc, ComputeType, scaleType));
#else
  checkCublasStatus(cublasLtMatmulDescCreate(&matmulDesc, ComputeType));
#endif
  checkCublasStatus(cublasLtMatmulDescSetAttribute(
      matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTrans, sizeof(opTrans)));

  float total_time = 0;
  for (int i = 0; i < iteration; ++i) {
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    int success = cublas_lt_matmul(handle, matmulDesc, XtransformDesc,
                                   WtransformDesc, YtransformDesc, Xtransform,
                                   Wtransform, Ytransform, alpha, beta);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (success > 0 && i >= 10) total_time += time;
  }
  if (total_time > 0) printf("%.3f ms\n", total_time / (iteration - 10));
  checkCublasStatus(cublasLtMatrixTransformDescSetAttribute(
      transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTrans,
      sizeof(opTrans)));
  checkCublasStatus(cublasLtMatrixTransform(
      handle, transformDesc, &transformAlpha, Ytransform, YtransformDesc,
      &transformBeta, NULL, NULL, Y, YDesc, 0));

  checkCublasStatus(cublasLtMatrixLayoutDestroy(XtransformDesc));
  checkCublasStatus(cublasLtMatrixLayoutDestroy(WtransformDesc));
  checkCublasStatus(cublasLtMatrixLayoutDestroy(YtransformDesc));
  checkCublasStatus(cublasLtMatrixLayoutDestroy(XDesc));
  checkCublasStatus(cublasLtMatrixLayoutDestroy(WDesc));
  checkCublasStatus(cublasLtMatrixLayoutDestroy(YDesc));
  checkCublasStatus(cublasLtMatmulDescDestroy(matmulDesc));
  checkCublasStatus(cublasLtMatrixTransformDescDestroy(transformDesc));
  cudaDeviceSynchronize();
  checkCudaStatus(cudaFree(Xtransform));
  checkCudaStatus(cudaFree(Wtransform));
  checkCudaStatus(cudaFree(Ytransform));
}

void _main(int C, int B, int O, int H, int iteration, bool debug) {
  // X:(C, B, H), W:(C, O, H), Y:(C, B, O)
  printf(
      ">>>>>>>>>>>>>>>>>>>> shape: X(%d, %d, %d), W(%d, %d, %d) "
      ">>>>>>>>>>>>>>>>>>>>\n",
      C, B, H, C, O, H);

  float *Y;
  if (debug) checkCudaStatus(cudaMallocManaged(&Y, C * B * O * sizeof(float)));

  float *fX, *fW, *fY;
  __half *hX, *hW, *hY;
  int8_t *iX, *iW;
  int32_t *iY;
  allocate_memory(C, B, O, H, &fX, &fW, &fY);
  allocate_memory(C, B, O, H, &hX, &hW, &hY);
  allocate_memory(C, B, O, H, &iX, &iW, &iY);

  float f_alpha = 1, f_beta = 0;
  __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
  int32_t i_alpha = 1, i_beta = 0;

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

  if (debug) matmul(fX, fW, Y, C, B, O, H);

  cublasLtHandle_t handle;
  checkCublasStatus(cublasLtCreate(&handle));

  printf(">>>>> test fp32 >>>>>\n");
  test_lt_matmul(handle, C, B, O, H, fX, fW, fY, &f_alpha, &f_beta, iteration);

  printf(">>>>> test fp16 >>>>>\n");
  test_lt_matmul(handle, C, B, O, H, hX, hW, hY, &h_alpha, &h_beta, iteration);

  printf(">>>>> test int8 >>>>>\n");
  test_lt_matmul_int8(handle, C, B, O, H, iX, iW, iY, &i_alpha, &i_beta,
                      iteration);

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

  free_memory(fX, fW, fY);
  free_memory(hX, hW, hY);
  free_memory(iX, iW, iY);
  if (debug) checkCudaStatus(cudaFree(Y));
}

int main() {
  int iteration = 50;
  bool debug = false;
  std::vector<int> Cs = {1, 8, 512};
  std::vector<int> Bs = {8, 16, 4096};
  std::vector<int> Os = {1024, 3072, 4096};
  std::vector<int> Hs = {1024, 4096};
  for (int l = 0; l < Cs.size(); ++l)
    for (int i = 0; i < Bs.size(); ++i)
      for (int j = 0; j < Os.size(); ++j)
        for (int k = 0; k < Hs.size(); ++k)
          _main(Cs[l], Bs[i], Os[j], Hs[k], iteration, debug);
  return 0;
}
