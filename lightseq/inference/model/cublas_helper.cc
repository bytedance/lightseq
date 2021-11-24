#include "cublas_helper.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>

#include "transformerKernels_int8.h"
#include "util.h"

namespace lightseq {
namespace cuda {

// for int8 cublasLtMM with algo
// ATransform should be m*n, CUBLASLT_ORDER_COL32
// kernel should be n*k, CUBLASLT_ORDER_COL4_4R2_8C or
// CUBLASLT_ORDER_COL32_2R_4R4 res is m*n, CUBLASLT_ORDER_COL32

void cublasLtMM_withAlgo(
    int* res, int batchCount, int m, int n, int k, int64_t stridea,
    int64_t strideb, int64_t stridec, const int8_t* ATransform,
    const int8_t* kernel, cublasLtHandle_t cublasLt_handle, cudaStream_t stream,
    // std::map<std::string, cublasLtMatmulAlgo_info>& cublasLtAlgoMap,
    bool use_ORDER_COL32_2R_4R4) {
  cublasOperation_t opTranspose = CUBLAS_OP_T;
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

  // create matmulDesc
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  cublasLtMatmulDescCreate(&matmulDesc, computeType, CUDA_R_32I);
#else
  cublasLtMatmulDescCreate(&matmulDesc, computeType);
#endif
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                 &opTranspose, sizeof(cublasOperation_t));
  cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, m, k, ldaTransform);
  cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_COL32, sizeof(order_COL32));
  cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, n, k, ldbTransform);
  cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_matrixB, sizeof(order_matrixB));
  cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_32I, m, n, ldcTransform);
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

  int alphaI = 1;
  int betaI = 0;

  // get algo
  cublasLtMatmulAlgo_t algo;
  char mark[1000];
  // sprintf(mark, "%d_%d_%d_%d_%d", batchCount, m, n, k, INT8_DATATYPE);
  std::string markStr(mark);
  int findAlgo = 0;
  //   if (cublasLtAlgoMap.find(markStr) != cublasLtAlgoMap.end() &&
  //   cublasLtAlgoMap[markStr].workspaceSize == 0)
  //   {
  //     //printf("find algo %s\n", markStr.c_str());
  //     findAlgo = 1;

  //     cublasLtMatmulAlgoInit(cublasLt_handle, computeType, CUDA_R_32I,
  //     CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUDA_R_32I,
  //     cublasLtAlgoMap[markStr].algoId, &algo);
  //     cublasLtMatmulAlgoConfigSetAttribute(&algo,
  //     CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
  //     &(cublasLtAlgoMap[markStr].customOption),
  //     sizeof(cublasLtAlgoMap[markStr].customOption));
  //     cublasLtMatmulAlgoConfigSetAttribute(&algo,
  //     CUBLASLT_ALGO_CONFIG_TILE_ID, &(cublasLtAlgoMap[markStr].tile),
  //     sizeof(cublasLtAlgoMap[markStr].tile));
  //     cublasLtMatmulAlgoConfigSetAttribute(&algo,
  //     CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
  //     &(cublasLtAlgoMap[markStr].splitK_val),
  //     sizeof(cublasLtAlgoMap[markStr].splitK_val));
  //     cublasLtMatmulAlgoConfigSetAttribute(&algo,
  //     CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
  //     &(cublasLtAlgoMap[markStr].swizzle),
  //     sizeof(cublasLtAlgoMap[markStr].swizzle));
  //     cublasLtMatmulAlgoConfigSetAttribute(&algo,
  //     CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
  //     &(cublasLtAlgoMap[markStr].reductionScheme), sizeof(int));
  // #if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  //     cublasLtMatmulAlgoConfigSetAttribute(&algo,
  //     CUBLASLT_ALGO_CONFIG_STAGES_ID, &(cublasLtAlgoMap[markStr].stages),
  //     sizeof(cublasLtAlgoMap[markStr].stages));
  // #endif
  //   }
  //   else
  //   {
  //     findAlgo = 1;
  //     int algoId;
  //     if (use_ORDER_COL32_2R_4R4)
  //     {
  //       algoId = 7;
  //     }
  //     else
  //     {
  //       algoId = 6;
  //     }
  //     int swizzle = 0;
  //     int customOption = 0;
  //     int tile = 20;
  //     int splitK_val = 0;
  //     int reductionScheme = 0;
  //     cublasLtMatmulAlgoInit(cublasLt_handle, computeType, CUDA_R_32I,
  //     CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUDA_R_32I, algoId, &algo);
  //     cublasLtMatmulAlgoConfigSetAttribute(&algo,
  //     CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(customOption),
  //     sizeof(customOption)); cublasLtMatmulAlgoConfigSetAttribute(&algo,
  //     CUBLASLT_ALGO_CONFIG_TILE_ID, &(tile), sizeof(tile));
  //     cublasLtMatmulAlgoConfigSetAttribute(&algo,
  //     CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(splitK_val), sizeof(splitK_val));
  //     cublasLtMatmulAlgoConfigSetAttribute(&algo,
  //     CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle), sizeof(swizzle));
  //     cublasLtMatmulAlgoConfigSetAttribute(&algo,
  //     CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &(reductionScheme),
  //     sizeof(int));
  // #if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  //     int stages;
  //     if (use_ORDER_COL32_2R_4R4)
  //       stages = 15;
  //     else
  //       stages = 13;
  //     cublasLtMatmulAlgoConfigSetAttribute(&algo,
  //     CUBLASLT_ALGO_CONFIG_STAGES_ID, &(stages), sizeof(stages));
  // #endif
  //   }

  findAlgo = 1;
  int algoId;
  if (use_ORDER_COL32_2R_4R4) {
    algoId = 7;
  } else {
    algoId = 6;
  }
  int swizzle = 0;
  int customOption = 0;
  int tile = 20;
  int splitK_val = 0;
  int reductionScheme = 0;
  cublasLtMatmulAlgoInit(cublasLt_handle, computeType, CUDA_R_32I, CUDA_R_8I,
                         CUDA_R_8I, CUDA_R_32I, CUDA_R_32I, algoId, &algo);
  cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                       CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                                       &(customOption), sizeof(customOption));
  cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID,
                                       &(tile), sizeof(tile));
  cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                       &(splitK_val), sizeof(splitK_val));
  cublasLtMatmulAlgoConfigSetAttribute(
      &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle), sizeof(swizzle));
  cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                       CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                       &(reductionScheme), sizeof(int));
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  int stages;
  if (use_ORDER_COL32_2R_4R4)
    stages = 15;
  else
    stages = 13;
  cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                                       &(stages), sizeof(stages));
#endif

  cublasLtMatmul(cublasLt_handle, matmulDesc, &alphaI, ATransform,
                 AtransformDesc, kernel, BtransformDesc, &betaI, res,
                 CtransformDesc, res, CtransformDesc,
                 (findAlgo == 1 ? (&algo) : NULL), NULL, 0, stream);

  cublasLtMatmulDescDestroy(matmulDesc);
  cublasLtMatrixLayoutDestroy(AtransformDesc);
  cublasLtMatrixLayoutDestroy(BtransformDesc);
  cublasLtMatrixLayoutDestroy(CtransformDesc);
}

// for int8 IO cublasLtMM with algo
// ATransform should be m*k CUBLASLT_ORDER_COL32
// kernel should be n*k CUBLASLT_ORDER_COL4_4R2_8C
// res is m*n CUBLASLT_ORDER_COL32
void cublasLtMM_withAlgo_int8IO(
    int8_t* res, int batchCount, int m, int n, int k, int64_t stridea,
    int64_t strideb, int64_t stridec, const float alpha,
    const int8_t* ATransform, const int8_t* kernel,
    cublasLtHandle_t cublasLt_handle, cudaStream_t stream,
    // std::map<std::string, cublasLtMatmulAlgo_info> &cublasLtAlgoMap,
    bool use_ORDER_COL32_2R_4R4) {
  cublasOperation_t opTranspose = CUBLAS_OP_T;
  // int8 gemm does not support CUBLAS_POINTER_MODE_DEVICE
  // cublasLtPointerMode_t pointerMode =
  // CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO;
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

  // create matmulDesc
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  cublasLtMatmulDescCreate(&matmulDesc, computeType, scaleType);
#else
  cublasLtMatmulDescCreate(&matmulDesc, computeType);
#endif
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                 &opTranspose, sizeof(cublasOperation_t));
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE,
                                 &scaleType, sizeof(scaleType));
  // cublasLtMatmulDescSetAttribute(matmulDesc,
  // CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointerMode,
  // sizeof(cublasLtPointerMode_t));
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
  // get algo
  cublasLtMatmulAlgo_t algo;
  char mark[1000];
  // sprintf(mark, "%d_%d_%d_%d_%d", batchCount, m, n, k, INT8_DATATYPE);
  std::string markStr(mark);
  int findAlgo = 0;
  //   if (cublasLtAlgoMap.find(markStr) != cublasLtAlgoMap.end() &&
  //       cublasLtAlgoMap[markStr].workspaceSize == 0) {
  //     findAlgo = 1;
  //     cublasLtMatmulAlgoInit(cublasLt_handle, computeType, CUDA_R_32F,
  //     CUDA_R_8I,
  //                            CUDA_R_8I, CUDA_R_8I, CUDA_R_8I,
  //                            cublasLtAlgoMap[markStr].algoId, &algo);
  //     cublasLtMatmulAlgoConfigSetAttribute(
  //         &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
  //         &(cublasLtAlgoMap[markStr].customOption),
  //         sizeof(cublasLtAlgoMap[markStr].customOption));
  //     cublasLtMatmulAlgoConfigSetAttribute(&algo,
  //     CUBLASLT_ALGO_CONFIG_TILE_ID,
  //                                          &(cublasLtAlgoMap[markStr].tile),
  //                                          sizeof(cublasLtAlgoMap[markStr].tile));
  //     cublasLtMatmulAlgoConfigSetAttribute(
  //         &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
  //         &(cublasLtAlgoMap[markStr].splitK_val),
  //         sizeof(cublasLtAlgoMap[markStr].splitK_val));
  //     cublasLtMatmulAlgoConfigSetAttribute(
  //         &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
  //         &(cublasLtAlgoMap[markStr].swizzle),
  //         sizeof(cublasLtAlgoMap[markStr].swizzle));
  //     cublasLtMatmulAlgoConfigSetAttribute(
  //         &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
  //         &(cublasLtAlgoMap[markStr].reductionScheme), sizeof(int));
  // #if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  //     cublasLtMatmulAlgoConfigSetAttribute(
  //         &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID,
  //         &(cublasLtAlgoMap[markStr].stages),
  //         sizeof(cublasLtAlgoMap[markStr].stages));
  // #endif
  //   } else {
  //     findAlgo = 1;
  //     int algoId;
  //     if (use_ORDER_COL32_2R_4R4) {
  //       algoId = 7;
  //     } else {
  //       algoId = 6;
  //     }
  //     int swizzle = 0;
  //     int customOption = 0;
  //     int tile = 20;
  //     int splitK_val = 0;
  //     int reductionScheme = 0;
  //     cublasLtMatmulAlgoInit(cublasLt_handle, computeType, CUDA_R_32F,
  //     CUDA_R_8I,
  //                            CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, algoId, &algo);
  //     cublasLtMatmulAlgoConfigSetAttribute(&algo,
  //                                          CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
  //                                          &(customOption),
  //                                          sizeof(customOption));
  //     cublasLtMatmulAlgoConfigSetAttribute(&algo,
  //     CUBLASLT_ALGO_CONFIG_TILE_ID,
  //                                          &(tile), sizeof(tile));
  //     cublasLtMatmulAlgoConfigSetAttribute(&algo,
  //     CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
  //                                          &(splitK_val),
  //                                          sizeof(splitK_val));
  //     cublasLtMatmulAlgoConfigSetAttribute(
  //         &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle),
  //         sizeof(swizzle));
  //     cublasLtMatmulAlgoConfigSetAttribute(&algo,
  //                                          CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
  //                                          &(reductionScheme), sizeof(int));
  // #if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  //     int stages;
  //     if (use_ORDER_COL32_2R_4R4)
  //       stages = 15;
  //     else
  //       stages = 13;
  //     cublasLtMatmulAlgoConfigSetAttribute(&algo,
  //     CUBLASLT_ALGO_CONFIG_STAGES_ID,
  //                                          &(stages), sizeof(stages));
  // #endif
  //   }

  findAlgo = 1;
  int algoId;
  if (use_ORDER_COL32_2R_4R4) {
    algoId = 7;
  } else {
    algoId = 6;
  }
  int swizzle = 0;
  int customOption = 0;
  int tile = 20;
  int splitK_val = 0;
  int reductionScheme = 0;
  cublasLtMatmulAlgoInit(cublasLt_handle, computeType, CUDA_R_32F, CUDA_R_8I,
                         CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, algoId, &algo);
  cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                       CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                                       &(customOption), sizeof(customOption));
  cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID,
                                       &(tile), sizeof(tile));
  cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                       &(splitK_val), sizeof(splitK_val));
  cublasLtMatmulAlgoConfigSetAttribute(
      &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle), sizeof(swizzle));
  cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                       CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                       &(reductionScheme), sizeof(int));
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  int stages;
  if (use_ORDER_COL32_2R_4R4)
    stages = 15;
  else
    stages = 13;
  cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                                       &(stages), sizeof(stages));
#endif

  float beta = 0.0f;
  CHECK_GPU_ERROR(cublasLtMatmul(
      cublasLt_handle, matmulDesc, &alpha, ATransform, AtransformDesc, kernel,
      BtransformDesc, &beta, res, CtransformDesc, res, CtransformDesc,
      (findAlgo == 1 ? (&algo) : NULL), NULL, 0, stream));

  CHECK_GPU_ERROR(cublasLtMatmulDescDestroy(matmulDesc));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(AtransformDesc));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(BtransformDesc));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(CtransformDesc));
}

void transform_weight_row_major2col32t(const int8_t* input, int8_t* output,
                                       int row, int col,
                                       cublasLtHandle_t lt_handle,
                                       cudaStream_t stream) {
  int ldtransform = 32 * roundoff(col, 8);
  float transform_alpha = 1.0f, transform_beta = 0.0f;
  cublasLtMatrixTransformDesc_t transform_desc = NULL;
  cublasLtMatrixLayout_t input_desc = NULL, output_desc = NULL;
  cublasLtOrder_t order_col = CUBLASLT_ORDER_COL;
  cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;
  //   cublasOperation_t opTranspose = CUBLAS_OP_T;

  CHECK_GPU_ERROR(
      cublasLtMatrixLayoutCreate(&input_desc, CUDA_R_8I, col, row, col));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(
      input_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_col, sizeof(order_col)));

  CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&output_desc, CUDA_R_8I, col, row,
                                             ldtransform));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(
      output_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL4_4R2_8C,
      sizeof(order_COL4_4R2_8C)));

  CHECK_GPU_ERROR(
      cublasLtMatrixTransformDescCreate(&transform_desc, CUDA_R_32F));
  //   CHECK_GPU_ERROR(cublasLtMatrixTransformDescSetAttribute(
  //       transform_desc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTranspose,
  //       sizeof(opTranspose)));

  CHECK_GPU_ERROR(cublasLtMatrixTransform(
      lt_handle, transform_desc, &transform_alpha, input, input_desc,
      &transform_beta, NULL, NULL, output, output_desc, stream));

  CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(input_desc));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(output_desc));
  CHECK_GPU_ERROR(cublasLtMatrixTransformDescDestroy(transform_desc));
}

template <typename T>
void quantize_weight_col32t(const T* origin_weight, int8_t* quantized_weight,
                            int rows, int cols, int quant_range, float clip_max,
                            cudaStream_t stream, cublasLtHandle_t handle) {
  int8_t* temp_weight;
  CHECK_GPU_ERROR(cudaMalloc(&temp_weight, rows * cols * sizeof(int8_t)));

  launch_quantize_tensor(origin_weight, temp_weight, rows, cols, quant_range,
                         clip_max, stream);
  CHECK_GPU_ERROR(cudaDeviceSynchronize());
  CHECK_GPU_ERROR(cudaGetLastError());

  transform_weight_row_major2col32t(temp_weight, quantized_weight, rows, cols,
                                    handle, stream);

  CHECK_GPU_ERROR(cudaFree(temp_weight));
}

template void quantize_weight_col32t<float>(const float* origin_weight,
                                            int8_t* quantized_weight, int rows,
                                            int cols, int quant_range,
                                            float clip_max, cudaStream_t stream,
                                            cublasLtHandle_t handle);

template void quantize_weight_col32t<half>(const half* origin_weight,
                                           int8_t* quantized_weight, int rows,
                                           int cols, int quant_range,
                                           float clip_max, cudaStream_t stream,
                                           cublasLtHandle_t handle);

}  // namespace cuda
}  // namespace lightseq
