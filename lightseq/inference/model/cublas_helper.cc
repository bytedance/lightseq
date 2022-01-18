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
// ATransform should be m*k, CUBLASLT_ORDER_COL32
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

  //   findAlgo = 1;
  //   int algoId;
  //   if (use_ORDER_COL32_2R_4R4) {
  //     algoId = 7;
  //   } else {
  //     algoId = 6;
  //   }
  //   int swizzle = 0;
  //   int customOption = 0;
  //   int tile = 20;
  //   int splitK_val = 0;
  //   int reductionScheme = 0;
  //   cublasLtMatmulAlgoInit(cublasLt_handle, computeType, CUDA_R_32I,
  //   CUDA_R_8I,
  //                          CUDA_R_8I, CUDA_R_32I, CUDA_R_32I, algoId, &algo);
  //   cublasLtMatmulAlgoConfigSetAttribute(&algo,
  //                                        CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
  //                                        &(customOption),
  //                                        sizeof(customOption));
  //   cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID,
  //                                        &(tile), sizeof(tile));
  //   cublasLtMatmulAlgoConfigSetAttribute(&algo,
  //   CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
  //                                        &(splitK_val), sizeof(splitK_val));
  //   cublasLtMatmulAlgoConfigSetAttribute(
  //       &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle),
  //       sizeof(swizzle));
  //   cublasLtMatmulAlgoConfigSetAttribute(&algo,
  //                                        CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
  //                                        &(reductionScheme), sizeof(int));
  // #if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  //   int stages;
  //   if (use_ORDER_COL32_2R_4R4)
  //     stages = 15;
  //   else
  //     stages = 13;
  //   cublasLtMatmulAlgoConfigSetAttribute(&algo,
  //   CUBLASLT_ALGO_CONFIG_STAGES_ID,
  //                                        &(stages), sizeof(stages));
  // #endif

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
void cublasLtMM_withAlgo_i8IO(
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

  //   findAlgo = 1;
  //   int algoId;
  //   if (use_ORDER_COL32_2R_4R4) {
  //     algoId = 7;
  //   } else {
  //     algoId = 6;
  //   }
  //   int swizzle = 0;
  //   int customOption = 0;
  //   int tile = 20;
  //   int splitK_val = 0;
  //   int reductionScheme = 0;
  //   cublasLtMatmulAlgoInit(cublasLt_handle, computeType, CUDA_R_32F,
  //   CUDA_R_8I,
  //                          CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, algoId, &algo);
  //   cublasLtMatmulAlgoConfigSetAttribute(&algo,
  //                                        CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
  //                                        &(customOption),
  //                                        sizeof(customOption));
  //   cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID,
  //                                        &(tile), sizeof(tile));
  //   cublasLtMatmulAlgoConfigSetAttribute(&algo,
  //   CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
  //                                        &(splitK_val), sizeof(splitK_val));
  //   cublasLtMatmulAlgoConfigSetAttribute(
  //       &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle),
  //       sizeof(swizzle));
  //   cublasLtMatmulAlgoConfigSetAttribute(&algo,
  //                                        CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
  //                                        &(reductionScheme), sizeof(int));
  // #if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  //   int stages;
  //   if (use_ORDER_COL32_2R_4R4)
  //     stages = 15;
  //   else
  //     stages = 13;
  //   cublasLtMatmulAlgoConfigSetAttribute(&algo,
  //   CUBLASLT_ALGO_CONFIG_STAGES_ID,
  //                                        &(stages), sizeof(stages));
  // #endif

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

template <typename OutType, typename ScaleType>
void cublaslt_gemm(const int8_t* input_a, const int8_t* input_b,
                   OutType* output_c, int batch_count, int m, int n, int k,
                   int64_t stridea, int64_t strideb, int64_t stridec,
                   const ScaleType alpha, cublasLtHandle_t cublasLt_handle,
                   cudaStream_t stream) {
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
  CHECK_GPU_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_dtype,
      sizeof(scale_dtype)));
#endif
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

  ScaleType beta = ScaleType(0);
  CHECK_GPU_ERROR(cublasLtMatmul(
      cublasLt_handle, matmul_desc, &alpha, input_a, desc_a, input_b, desc_b,
      &beta, output_c, desc_c, output_c, desc_c, NULL, NULL, 0, stream));

  CHECK_GPU_ERROR(cublasLtMatmulDescDestroy(matmul_desc));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(desc_a));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(desc_b));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(desc_c));
}

template void cublaslt_gemm<int32_t, int32_t>(
    const int8_t* input_a, const int8_t* input_b, int32_t* output_c,
    int batchCount, int m, int n, int k, int64_t stridea, int64_t strideb,
    int64_t stridec, const int32_t alpha, cublasLtHandle_t cublasLt_handle,
    cudaStream_t stream);

template void cublaslt_gemm<int8_t, float>(
    const int8_t* input_a, const int8_t* input_b, int8_t* output_c,
    int batchCount, int m, int n, int k, int64_t stridea, int64_t strideb,
    int64_t stridec, const float alpha, cublasLtHandle_t cublasLt_handle,
    cudaStream_t stream);

void transform_weight_layout(const int8_t* input, int8_t* output, int row,
                             int col, Layout layout, cublasLtHandle_t lt_handle,
                             cudaStream_t stream) {
  float transform_alpha = 1.0f, transform_beta = 0.0f;
  cublasLtMatrixTransformDesc_t transform_desc = NULL;
  cublasLtMatrixLayout_t input_desc = NULL, output_desc = NULL;
  cublasLtOrder_t order_col = CUBLASLT_ORDER_COL;
  cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;
  cublasOperation_t transpose = CUBLAS_OP_T;

  CHECK_GPU_ERROR(
      cublasLtMatrixLayoutCreate(&input_desc, CUDA_R_8I, col, row, col));
  CHECK_GPU_ERROR(
      cublasLtMatrixTransformDescCreate(&transform_desc, CUDA_R_32F));

  if (layout == kColMajor32) {
    int ldtransform = 32 * round_up(col, 8);
    CHECK_GPU_ERROR(cublasLtMatrixLayoutCreate(&output_desc, CUDA_R_8I, col,
                                               row, ldtransform));
    CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(
        output_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL4_4R2_8C,
        sizeof(order_COL4_4R2_8C)));
  } else if (layout == kColMajor) {
    CHECK_GPU_ERROR(
        cublasLtMatrixLayoutCreate(&output_desc, CUDA_R_8I, row, col, row));
    CHECK_GPU_ERROR(cublasLtMatrixTransformDescSetAttribute(
        transform_desc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &transpose,
        sizeof(transpose)));
  } else {
    throw std::runtime_error("unsupported layout");
  }

  CHECK_GPU_ERROR(cublasLtMatrixTransform(
      lt_handle, transform_desc, &transform_alpha, input, input_desc,
      &transform_beta, NULL, NULL, output, output_desc, stream));

  CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(input_desc));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(output_desc));
  CHECK_GPU_ERROR(cublasLtMatrixTransformDescDestroy(transform_desc));
}

template <typename T>
void quantize_weight(const T* origin_weight, int8_t* quantized_weight, int rows,
                     int cols, float quant_scale, cudaStream_t stream,
                     cublasLtHandle_t handle, bool layout_col32t) {
  int8_t* temp_weight;
  CHECK_GPU_ERROR(cudaMalloc(&temp_weight, rows * cols * sizeof(int8_t)));

  launch_quantize_tensor(origin_weight, temp_weight, rows, cols, quant_scale,
                         stream);
  CHECK_GPU_ERROR(cudaDeviceSynchronize());
  CHECK_GPU_ERROR(cudaGetLastError());

  if (layout_col32t) {
    transform_weight_layout(temp_weight, quantized_weight, rows, cols,
                            kColMajor32, handle, stream);
  } else {
    transform_weight_layout(temp_weight, quantized_weight, rows, cols,
                            kColMajor, handle, stream);
  }
  CHECK_GPU_ERROR(cudaFree(temp_weight));
}

template void quantize_weight<float>(const float* origin_weight,
                                     int8_t* quantized_weight, int rows,
                                     int cols, float quant_scale,
                                     cudaStream_t stream,
                                     cublasLtHandle_t handle,
                                     bool layout_col32t);

template void quantize_weight<half>(const half* origin_weight,
                                    int8_t* quantized_weight, int rows,
                                    int cols, float quant_scale,
                                    cudaStream_t stream,
                                    cublasLtHandle_t handle,
                                    bool layout_col32t);

}  // namespace cuda
}  // namespace lightseq
