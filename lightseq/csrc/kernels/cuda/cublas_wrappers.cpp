/* Copyright 2021 The LightSeq Team
   Copyright Microsoft DeepSpeed
   This file is adapted from Microsoft DeepSpeed
*/
#include "cublas_wrappers.h"

#include <stdexcept>

#include "cuda_util.h"
namespace lightseq {
namespace cuda {
int cublas_gemm_ex(cublasHandle_t handle, cublasOperation_t transa,
                   cublasOperation_t transb, int m, int n, int k,
                   const float *alpha, const float *beta, const float *A,
                   const float *B, float *C, cublasGemmAlgo_t algo) {
  cublasStatus_t status =
      cublasGemmEx(handle, transa, transb, m, n, k, (const void *)alpha,
                   (const void *)A, CUDA_R_32F, (transa == CUBLAS_OP_N) ? m : k,
                   (const void *)B, CUDA_R_32F, (transb == CUBLAS_OP_N) ? k : n,
                   (const void *)beta, C, CUDA_R_32F, m, CUDA_R_32F, algo);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr,
            "!!!! kernel cublasGemmEx(float*) execution error. (m: %d, n: %d, "
            "k: %d, error: %d) \n",
            m, n, k, (int)status);
    return EXIT_FAILURE;
  }
  return 0;
}

int cublas_gemm_ex(cublasHandle_t handle, cublasOperation_t transa,
                   cublasOperation_t transb, int m, int n, int k,
                   const float *alpha, const float *beta, const __half *A,
                   const __half *B, __half *C, cublasGemmAlgo_t algo) {
  cublasStatus_t status = cublasGemmEx(
      handle, transa, transb, m, n, k, (const void *)alpha, (const void *)A,
      CUDA_R_16F, (transa == CUBLAS_OP_N) ? m : k, (const void *)B, CUDA_R_16F,
      (transb == CUBLAS_OP_N) ? k : n, (const void *)beta, (void *)C,
      CUDA_R_16F, m, CUDA_R_32F, algo);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr,
            "!!!! kernel cublasGemmEx(__half*) execution error. (m: %d, n: %d, "
            "k: %d, error: %d) \n",
            m, n, k, (int)status);
    return EXIT_FAILURE;
  }
  return 0;
}

int cublas_strided_batched_gemm(cublasHandle_t handle, int m, int n, int k,
                                const float *alpha, const float *beta,
                                const float *A, const float *B, float *C,
                                cublasOperation_t op_A, cublasOperation_t op_B,
                                int stride_A, int stride_B, int stride_C,
                                int batch, cublasGemmAlgo_t algo) {
  cublasStatus_t status = cublasGemmStridedBatchedEx(
      handle, op_A, op_B, m, n, k, alpha, A, CUDA_R_32F,
      (op_A == CUBLAS_OP_N) ? m : k, stride_A, B, CUDA_R_32F,
      (op_B == CUBLAS_OP_N) ? k : n, stride_B, beta, C, CUDA_R_32F, m, stride_C,
      batch, CUDA_R_32F, algo);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr,
            "!!!! kernel cublasGemmStridedBatchedEx(float*) execution error. "
            "(batch: %d, m: %d, n: %d, k: %d, "
            "error: %d) \n",
            batch, m, n, k, (int)status);
    return EXIT_FAILURE;
  }
  return 0;
}

int cublas_strided_batched_gemm(cublasHandle_t handle, int m, int n, int k,
                                const float *alpha, const float *beta,
                                const __half *A, const __half *B, __half *C,
                                cublasOperation_t op_A, cublasOperation_t op_B,
                                int stride_A, int stride_B, int stride_C,
                                int batch, cublasGemmAlgo_t algo) {
  cublasStatus_t status = cublasGemmStridedBatchedEx(
      handle, op_A, op_B, m, n, k, alpha, A, CUDA_R_16F,
      (op_A == CUBLAS_OP_N) ? m : k, stride_A, B, CUDA_R_16F,
      (op_B == CUBLAS_OP_N) ? k : n, stride_B, beta, C, CUDA_R_16F, m, stride_C,
      batch, CUDA_R_32F, algo);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr,
            "!!!! kernel cublasGemmStridedBatchedEx(__half*) execution error. "
            "(m: %d, n: %d, k: %d, error: %d) \n",
            m, n, k, (int)status);
    return EXIT_FAILURE;
  }

  return 0;
}

template <typename OutType, typename ScaleType>
void cublaslt_igemm(const int8_t *input_a, const int8_t *input_b,
                    OutType *output_c, int batch_count, int m, int n, int k,
                    int64_t stridea, int64_t strideb, int64_t stridec,
                    const ScaleType *alpha, const ScaleType *beta,
                    cublasLtHandle_t cublasLt_handle, cudaStream_t stream) {
  cublasOperation_t transpose = CUBLAS_OP_T;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_32I;
#else
  cudaDataType_t compute_type = CUDA_R_32I;
#endif
  cublasLtMatmulDesc_t matmul_desc;
  cublasLtMatrixLayout_t desc_a = NULL;
  cublasLtMatrixLayout_t desc_b = NULL;
  cublasLtMatrixLayout_t desc_c = NULL;

  cudaDataType_t out_dtype;
  cudaDataType_t scale_dtype;
  if (std::is_same<OutType, int32_t>::value) {
    out_dtype = CUDA_R_32I;
    scale_dtype = CUDA_R_32I;
  } else if (std::is_same<OutType, int8_t>::value) {
    out_dtype = CUDA_R_8I;
    scale_dtype = CUDA_R_32F;
  } else {
    throw std::runtime_error("Unsupported output type");
  }

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  CHECK_GPU_ERROR(
      cublasLtMatmulDescCreate(&matmul_desc, compute_type, scale_dtype));
#else
  CHECK_GPU_ERROR(cublasLtMatmulDescCreate(&matmul_desc, compute_type));
#endif

  cublasLtPointerMode_t scale_mode = CUBLASLT_POINTER_MODE_DEVICE;
  CHECK_GPU_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_dtype,
      sizeof(scale_dtype)));
  CHECK_GPU_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &scale_mode,
      sizeof(scale_mode)));
  CHECK_GPU_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));

  CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&desc_a, CUDA_R_8I, k, m, k));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&desc_b, CUDA_R_8I, k, n, k));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&desc_c, out_dtype, m, n, m));

  if (batch_count > 1) {
    CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(
        desc_a, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
        sizeof(batch_count)));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(
        desc_a, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea,
        sizeof(stridea)));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(
        desc_b, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
        sizeof(batch_count)));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(
        desc_b, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb,
        sizeof(strideb)));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(
        desc_c, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
        sizeof(batch_count)));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(
        desc_c, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec,
        sizeof(stridec)));
  }

  CHECK_GPU_ERROR(cublasLtMatmul(
      cublasLt_handle, matmul_desc, alpha, input_a, desc_a, input_b, desc_b,
      beta, output_c, desc_c, output_c, desc_c, NULL, NULL, 0, stream));

  CHECK_GPU_ERROR(cublasLtMatmulDescDestroy(matmul_desc));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(desc_a));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(desc_b));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(desc_c));
}

template void cublaslt_igemm<int32_t, int32_t>(
    const int8_t *input_a, const int8_t *input_b, int32_t *output_c,
    int batch_count, int m, int n, int k, int64_t stridea, int64_t strideb,
    int64_t stridec, const int32_t *alpha, const int32_t *beta,
    cublasLtHandle_t cublasLt_handle, cudaStream_t stream);

template void cublaslt_igemm<int8_t, float>(
    const int8_t *input_a, const int8_t *input_b, int8_t *output_c,
    int batch_count, int m, int n, int k, int64_t stridea, int64_t strideb,
    int64_t stridec, const float *alpha, const float *beta,
    cublasLtHandle_t cublasLt_handle, cudaStream_t stream);

template <typename OutType, typename ScaleType>
void cublaslt_igemm(const int8_t *input_a, const int8_t *input_b,
                    OutType *output_c, int batch_count, int m, int n, int k,
                    int64_t stridea, int64_t strideb, int64_t stridec,
                    const ScaleType *alpha, const ScaleType *beta,
                    cublasLtHandle_t cublasLt_handle, cudaStream_t stream,
                    cublasLtMatmulAlgo_info &algo_info,
                    cublasAlgoMap &algo_map) {
  cublasOperation_t transpose = CUBLAS_OP_T;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_32I;
#else
  cudaDataType_t compute_type = CUDA_R_32I;
#endif
  cublasLtMatmulDesc_t matmul_desc;
  cublasLtMatrixLayout_t desc_a = NULL;
  cublasLtMatrixLayout_t desc_b = NULL;
  cublasLtMatrixLayout_t desc_c = NULL;

  cudaDataType_t out_dtype;
  cudaDataType_t scale_dtype;
  if (std::is_same<OutType, int32_t>::value) {
    out_dtype = CUDA_R_32I;
    scale_dtype = CUDA_R_32I;
  } else if (std::is_same<OutType, int8_t>::value) {
    out_dtype = CUDA_R_8I;
    scale_dtype = CUDA_R_32F;
  } else {
    throw std::runtime_error("Unsupported output type");
  }

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  CHECK_GPU_ERROR(
      cublasLtMatmulDescCreate(&matmul_desc, compute_type, scale_dtype));
#else
  CHECK_GPU_ERROR(cublasLtMatmulDescCreate(&matmul_desc, compute_type));
#endif

  cublasLtPointerMode_t scale_mode = CUBLASLT_POINTER_MODE_DEVICE;
  CHECK_GPU_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_dtype,
      sizeof(scale_dtype)));
  CHECK_GPU_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &scale_mode,
      sizeof(scale_mode)));
  CHECK_GPU_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));

  CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&desc_a, CUDA_R_8I, k, m, k));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&desc_b, CUDA_R_8I, k, n, k));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&desc_c, out_dtype, m, n, m));

  if (batch_count > 1) {
    CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(
        desc_a, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
        sizeof(batch_count)));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(
        desc_a, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea,
        sizeof(stridea)));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(
        desc_b, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
        sizeof(batch_count)));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(
        desc_b, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb,
        sizeof(strideb)));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(
        desc_c, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
        sizeof(batch_count)));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(
        desc_c, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec,
        sizeof(stridec)));
  }

  if (algo_info.algoId != -1) {
    cublasLtMatmulAlgo_t algo;
    cublasLtMatmulAlgoInit(cublasLt_handle, compute_type, CUDA_R_32F, CUDA_R_8I,
                           CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, algo_info.algoId,
                           &algo);
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(algo_info.customOption),
        sizeof(algo_info.customOption));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID,
                                         &(algo_info.tile),
                                         sizeof(algo_info.tile));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                         &(algo_info.splitK_val),
                                         sizeof(algo_info.splitK_val));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(algo_info.swizzle),
        sizeof(algo_info.swizzle));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
        &(algo_info.reductionScheme), sizeof(algo_info.reductionScheme));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                                         &(algo_info.stages),
                                         sizeof(algo_info.stages));

    CHECK_GPU_ERROR(cublasLtMatmul(
        cublasLt_handle, matmul_desc, alpha, input_a, desc_a, input_b, desc_b,
        beta, output_c, desc_c, output_c, desc_c, &algo,
        algo_map.get_workspace(), algo_map.get_workspace_size(), stream));
  } else {
    CHECK_GPU_ERROR(cublasLtMatmul(
        cublasLt_handle, matmul_desc, alpha, input_a, desc_a, input_b, desc_b,
        beta, output_c, desc_c, output_c, desc_c, nullptr, nullptr, 0, stream));
  }

  CHECK_GPU_ERROR(cublasLtMatmulDescDestroy(matmul_desc));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(desc_a));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(desc_b));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(desc_c));
}

template void cublaslt_igemm<int32_t, int32_t>(
    const int8_t *input_a, const int8_t *input_b, int32_t *output_c,
    int batch_count, int m, int n, int k, int64_t stridea, int64_t strideb,
    int64_t stridec, const int32_t *alpha, const int32_t *beta,
    cublasLtHandle_t cublasLt_handle, cudaStream_t stream,
    cublasLtMatmulAlgo_info &algo_info, cublasAlgoMap &algo_map);

template void cublaslt_igemm<int8_t, float>(
    const int8_t *input_a, const int8_t *input_b, int8_t *output_c,
    int batch_count, int m, int n, int k, int64_t stridea, int64_t strideb,
    int64_t stridec, const float *alpha, const float *beta,
    cublasLtHandle_t cublasLt_handle, cudaStream_t stream,
    cublasLtMatmulAlgo_info &algo_info, cublasAlgoMap &algo_map);

/**
 * @brief cublasLt imma gemm for i8 in i8 out
 *
 * @param res
 * @param batchCount
 * @param m
 * @param n
 * @param k
 * @param stridea
 * @param strideb
 * @param stridec
 * @param alpha
 * @param ATransform
 * @param kernel
 * @param cublasLt_handle
 * @param stream
 * @param use_ORDER_COL32_2R_4R4
 */
void cublasLtMM_withAlgo_i8IO(int8_t *res, int batchCount, int m, int n, int k,
                              int64_t stridea, int64_t strideb, int64_t stridec,
                              const float *alpha, const float *beta,
                              const int8_t *ATransform, const int8_t *kernel,
                              cublasLtHandle_t cublasLt_handle,
                              cudaStream_t stream,
                              bool use_ORDER_COL32_2R_4R4) {
  cublasOperation_t opTranspose = CUBLAS_OP_T;
  cudaDataType_t scaleType = CUDA_R_32F;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
#else
  cudaDataType_t computeType = CUDA_R_32I;
#endif
  cublasLtMatmulDesc_t matmulDesc;
  cublasLtMatrixLayout_t AtransformDesc = NULL;
  cublasLtMatrixLayout_t BtransformDesc = NULL;
  cublasLtMatrixLayout_t CtransformDesc = NULL;
  cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;

  cublasLtOrder_t order_matrixB;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  if (use_ORDER_COL32_2R_4R4)
    order_matrixB = CUBLASLT_ORDER_COL32_2R_4R4;
  else
    order_matrixB = CUBLASLT_ORDER_COL4_4R2_8C;
#else
  order_matrixB = CUBLASLT_ORDER_COL4_4R2_8C;
#endif

  int ldaTransform = 32 * m;

  int ldbTransform;
  if (use_ORDER_COL32_2R_4R4)
    ldbTransform = 32 * ((n + 32 - 1) / 32) * 32;
  else
    ldbTransform = 32 * ((n + 8 - 1) / 8) * 8;

  int ldcTransform = 32 * m;

  cublasLtPointerMode_t scale_mode =
      CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO;

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  cublasLtMatmulDescCreate(&matmulDesc, computeType, scaleType);
#else
  cublasLtMatmulDescCreate(&matmulDesc, computeType);
#endif
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                 &opTranspose, sizeof(cublasOperation_t));
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE,
                                 &scaleType, sizeof(scaleType));
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                 &scale_mode, sizeof(scale_mode));

  cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, m, k, ldaTransform);
  cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_COL32, sizeof(order_COL32));
  cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, n, k, ldbTransform);
  cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_matrixB, sizeof(order_matrixB));
  cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_8I, m, n, ldcTransform);
  cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_COL32, sizeof(order_COL32));
  if (batchCount > 1) {
    cublasLtMatrixLayoutSetAttribute(AtransformDesc,
                                     CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                     &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(
        AtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea,
        sizeof(stridea));
    cublasLtMatrixLayoutSetAttribute(BtransformDesc,
                                     CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                     &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(
        BtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb,
        sizeof(strideb));
    cublasLtMatrixLayoutSetAttribute(CtransformDesc,
                                     CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                     &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(
        CtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec,
        sizeof(stridec));
  }

  CHECK_GPU_ERROR(cublasLtMatmul(cublasLt_handle, matmulDesc, alpha, ATransform,
                                 AtransformDesc, kernel, BtransformDesc, beta,
                                 res, CtransformDesc, res, CtransformDesc, NULL,
                                 NULL, 0, stream));

  CHECK_GPU_ERROR(cublasLtMatmulDescDestroy(matmulDesc));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(AtransformDesc));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(BtransformDesc));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(CtransformDesc));
}

void cublasLtMM_withAlgo_i8IO(int8_t *res, int batchCount, int m, int n, int k,
                              int64_t stridea, int64_t strideb, int64_t stridec,
                              const float *alpha, const float *beta,
                              const int8_t *ATransform, const int8_t *kernel,
                              cublasLtHandle_t cublasLt_handle,
                              cudaStream_t stream,
                              cublasLtMatmulAlgo_info &algo_info,
                              cublasAlgoMap &algo_map) {
  cublasOperation_t opTranspose = CUBLAS_OP_T;
  cudaDataType_t scaleType = CUDA_R_32F;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
#else
  cudaDataType_t computeType = CUDA_R_32I;
#endif
  cublasLtMatmulDesc_t matmulDesc;
  cublasLtMatrixLayout_t AtransformDesc = NULL;
  cublasLtMatrixLayout_t BtransformDesc = NULL;
  cublasLtMatrixLayout_t CtransformDesc = NULL;
  cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;

  cublasLtOrder_t order_matrixB;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  if (algo_info.dataOrder == "CUBLASLT_ORDER_COL32_2R_4R4")
    order_matrixB = CUBLASLT_ORDER_COL32_2R_4R4;
  else
    order_matrixB = CUBLASLT_ORDER_COL4_4R2_8C;
#else
  order_matrixB = CUBLASLT_ORDER_COL4_4R2_8C;
#endif

  int ldaTransform = 32 * m;

  int ldbTransform;
  if (algo_info.dataOrder == "CUBLASLT_ORDER_COL32_2R_4R4")
    ldbTransform = 32 * ((n + 32 - 1) / 32) * 32;
  else
    ldbTransform = 32 * ((n + 8 - 1) / 8) * 8;

  int ldcTransform = 32 * m;

  cublasLtPointerMode_t scale_mode =
      CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO;

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  cublasLtMatmulDescCreate(&matmulDesc, computeType, scaleType);
#else
  cublasLtMatmulDescCreate(&matmulDesc, computeType);
#endif
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                 &opTranspose, sizeof(cublasOperation_t));
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE,
                                 &scaleType, sizeof(scaleType));
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                 &scale_mode, sizeof(scale_mode));

  cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, m, k, ldaTransform);
  cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_COL32, sizeof(order_COL32));
  cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, n, k, ldbTransform);
  cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_matrixB, sizeof(order_matrixB));
  cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_8I, m, n, ldcTransform);
  cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_COL32, sizeof(order_COL32));
  if (batchCount > 1) {
    cublasLtMatrixLayoutSetAttribute(AtransformDesc,
                                     CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                     &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(
        AtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea,
        sizeof(stridea));
    cublasLtMatrixLayoutSetAttribute(BtransformDesc,
                                     CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                     &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(
        BtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb,
        sizeof(strideb));
    cublasLtMatrixLayoutSetAttribute(CtransformDesc,
                                     CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                     &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(
        CtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec,
        sizeof(stridec));
  }

  if (algo_info.algoId != -1) {
    cublasLtMatmulAlgo_t algo;
    cublasLtMatmulAlgoInit(cublasLt_handle, computeType, CUDA_R_32F, CUDA_R_8I,
                           CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, algo_info.algoId,
                           &algo);
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(algo_info.customOption),
        sizeof(algo_info.customOption));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID,
                                         &(algo_info.tile),
                                         sizeof(algo_info.tile));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                         &(algo_info.splitK_val),
                                         sizeof(algo_info.splitK_val));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(algo_info.swizzle),
        sizeof(algo_info.swizzle));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
        &(algo_info.reductionScheme), sizeof(algo_info.reductionScheme));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                                         &(algo_info.stages),
                                         sizeof(algo_info.stages));

    CHECK_GPU_ERROR(cublasLtMatmul(
        cublasLt_handle, matmulDesc, alpha, ATransform, AtransformDesc, kernel,
        BtransformDesc, beta, res, CtransformDesc, res, CtransformDesc, &algo,
        algo_map.get_workspace(), algo_map.get_workspace_size(), stream));
  } else {
    CHECK_GPU_ERROR(cublasLtMatmul(cublasLt_handle, matmulDesc, alpha,
                                   ATransform, AtransformDesc, kernel,
                                   BtransformDesc, beta, res, CtransformDesc,
                                   res, CtransformDesc, NULL, NULL, 0, stream));
  }

  CHECK_GPU_ERROR(cublasLtMatmulDescDestroy(matmulDesc));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(AtransformDesc));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(BtransformDesc));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(CtransformDesc));
}
}  // namespace cuda
}  // namespace lightseq
