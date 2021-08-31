#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include "util.h"

template <typename T, typename S>
int cublas_gemm_ex(cublasHandle_t handle, cublasOperation_t transA,
                   cublasOperation_t transB, int m, int n, int k, T *A, T *B,
                   S *C, int lda, int ldb, int ldc, cudaDataType_t AType,
                   cudaDataType_t BType, cudaDataType_t CType,
                   cudaDataType_t ComputeType, S *alpha, S *beta,
                   cublasGemmAlgo_t algo) {
  cublasStatus_t status;
  status = cublasGemmEx(handle, transA, transB, m, n, k, alpha, A, AType, lda,
                        B, BType, ldb, beta, C, CType, ldc, ComputeType, algo);

  if (status == CUBLAS_STATUS_SUCCESS)
    return 1;
  else
    return -1;
}

template <typename T, typename S>
int cublas_gemm_strided_batched_ex(
    cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB,
    int bsz, int m, int n, int k, T *A, T *B, S *C, int lda, int ldb, int ldc,
    cudaDataType_t AType, cudaDataType_t BType, cudaDataType_t CType,
    cudaDataType_t ComputeType, S *alpha, S *beta, cublasGemmAlgo_t algo) {
  cublasStatus_t status;
  int64_t strideA = m * k, strideB = k * n, strideC = m * n;
  status = cublasGemmStridedBatchedEx(
      handle, transA, transB, m, n, k, alpha, A, AType, lda, strideA, B, BType,
      ldb, strideB, beta, C, CType, ldc, strideC, bsz, ComputeType, algo);

  if (status == CUBLAS_STATUS_SUCCESS)
    return 1;
  else
    return -1;
}

template <typename T, typename S>
void test_gemm_ex(cublasHandle_t handle, int C, int B, int O, int H, T *X, T *W,
                  S *Y, S *alpha, S *beta, int algo, int iteration) {
  cudaDataType_t AType, BType, CType, ComputeType;
  if (std::is_same<T, float>::value) {
    AType = BType = CType = ComputeType = CUDA_R_32F;
  } else if (std::is_same<T, __half>::value) {
    AType = BType = CType = ComputeType = CUDA_R_16F;
  } else if (std::is_same<T, int8_t>::value) {
    AType = BType = CUDA_R_8I;
    CType = ComputeType = CUDA_R_32I;
  } else {
    printf("Not supported data type.");
    return;
  }

  float total_time = 0;
  for (int i = 0; i < iteration; ++i) {
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    int success;
    if (C > 1)
      cublas_gemm_strided_batched_ex(handle, CUBLAS_OP_T, CUBLAS_OP_N, C, O, B,
                                     H, W, X, Y, H, H, O, AType, BType, CType,
                                     ComputeType, alpha, beta,
                                     static_cast<cublasGemmAlgo_t>(algo));
    else
      cublas_gemm_ex(handle, CUBLAS_OP_T, CUBLAS_OP_N, O, B, H, W, X, Y, H, H,
                     O, AType, BType, CType, ComputeType, alpha, beta,
                     static_cast<cublasGemmAlgo_t>(algo));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (success > 0 && i >= 10) total_time += time;
  }
  if (total_time > 0)
    printf("algo %d: %.3f ms\n", algo, total_time / (iteration - 10));
}

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
  if (C > 1) {
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
  }

  T *Wtransform;
  checkCudaStatus(cudaMalloc(reinterpret_cast<void **>(&Wtransform),
                             sizeof(T) * C * O * H));

  cublasLtMatrixLayout_t WtransformDesc;
  checkCublasStatus(
      cublasLtMatrixLayoutCreate(&WtransformDesc, WType, O, H, O));
  if (C > 1) {
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
        WtransformDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &C, sizeof(C)));
    checkCublasStatus(cublasLtMatrixLayoutSetAttribute(
        WtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideW,
        sizeof(strideW)));
  }

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
  if (C > 1) {
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
  }

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
  if (C > 1) {
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
  }

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
