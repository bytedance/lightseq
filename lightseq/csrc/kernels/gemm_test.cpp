/* Copyright 2021 The LightSeq Team
   Copyright NVIDIA FasterTransformer
   This file is adapted from NVIDIA FasterTransformer

This is the sample code to profile :
    1. FP16 NN-gemm + cublas (using tensor core);
    2. INT8 NT-gemm + cublasLt + INT8 Output. (using tensor core + IMMA specific
layout) (using alpha to quantize)
    3. INT8 TN-gemm + cublasLt + INT8 Output. (using tensor core + col-major
layout) (using alpha to quantize)

Find the best algorithms first, and then do the profiling.
*/
#include <iostream>
#include <map>
#include <unistd.h>
using namespace std;

#include <algorithm>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
// cuda
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "kernels.h"

#define Value 127
#define checkCudaAPIErrors(F)                                       \
  if ((F) != cudaSuccess) {                                         \
    printf("Error at line %d in file %s: %s\n", __LINE__, __FILE__, \
           cudaGetErrorString(cudaGetLastError()));                 \
    exit(-1);                                                       \
  }

int RUN_WITH_A100 = 1;
string gemm_test_result = "";
char tmp_cstr[1024];
string tmp_str;

// mat is column-major
template <typename T>
void transpose(T* matT, T* mat, int rows, int cols) {
  for (int c = 0; c < cols; c++) {
    for (int r = 0; r < rows; r++) {
      int indexIn = r + c * rows;
      int indexOut = c + r * cols;

      matT[indexOut] = mat[indexIn];
    }
  }
}

template <typename T>
void matMul(int m, int n, int k, const T* A, int lda, const T* B, int ldb,
            float* C, int ldc) {
  float sum;
  for (int c = 0; c < n; c++) {
    // printf("CPU gemm : %f\n", float(c)/float(n));
    for (int r = 0; r < m; r++) {
      sum = 0;
      for (int kk = 0; kk < k; kk++) {
        int idxA = kk * lda + r;  // A[r][kk]
        int idxB = c * ldb + kk;  // B[kk][c]
        sum += float(A[idxA]) * float(B[idxB]);
      }
      C[c * ldc + r] = sum;  // C[r][c]
    }
  }
}

static const char* _cudaGetErrorEnum(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }

  return "<unknown>";
}

#define checkcuBlasError(F)                                         \
  if ((F) != CUBLAS_STATUS_SUCCESS) {                               \
    printf("Error at line %d in file %s: %s\n", __LINE__, __FILE__, \
           _cudaGetErrorEnum(F));                                   \
    exit(-1);                                                       \
  }

int roundoff(int v, int d) { return ((v + d - 1) / d) * d; }

// initialize matrix
void matInit(half* p, int8_t* p2, int size) {
  srand(time(NULL));
  for (int index = 0; index < size; index++) {
    float tmp = rand() % 10 - 5;
    p2[index] = int8_t(tmp);
    p[index] = half(tmp);
  }
}

template <typename T>
int checkNonZero(T* A, int size) {
  T* h_A = (T*)malloc(sizeof(T) * size);
  cudaMemcpy(h_A, A, sizeof(T) * size, cudaMemcpyDeviceToHost);
  int noneZeroNum = 0;
  for (int ii = 0; ii < size; ii++) {
    if (fabs(float(h_A[ii]) - 0.0f) > 0.0001f) {
      noneZeroNum += 1;
    }
  }
  free(h_A);
  return noneZeroNum;
}

template <typename TA, typename TB>
void checkMat(TA* A, TB* B, int size) {
  TA* matA = (TA*)malloc(sizeof(TA) * size);
  TB* matB = (TB*)malloc(sizeof(TB) * size);
  int not_passed = 0;
  cudaMemcpy(matA, A, sizeof(TA) * size, cudaMemcpyDeviceToHost);
  cudaMemcpy(matB, B, sizeof(TB) * size, cudaMemcpyDeviceToHost);
  float A_nonZero_ratio = float(checkNonZero(A, size)) / float(size);
  float B_nonZero_ratio = float(checkNonZero(B, size)) / float(size);
  if (A_nonZero_ratio < 0.1 || B_nonZero_ratio < 0.1)
    printf("[checkMat] nonZero ratio [%f] [%f]\n", A_nonZero_ratio,
           B_nonZero_ratio);
  for (int jjj = 0; jjj < size; jjj++) {
    if (fabs(float(matA[jjj]) - float(matB[jjj])) > 0.00001) {
      not_passed += 1;
      printf("%d %f %f %f\n", jjj, float(matA[jjj]), float(matB[jjj]),
             float(matA[jjj]) - float(matB[jjj]));
    }
  }
  if (not_passed != 0)
    printf("[checkMat] different elements : %d \n", not_passed);
  free(matA);
  free(matB);
}

template <typename T>
int checkNonZero2(T* A, int size) {
  T* h_A = A;
  int noneZeroNum = 0;
  for (int ii = 0; ii < size; ii++) {
    if (fabs(float(h_A[ii]) - 0.0f) > 0.0001f) {
      noneZeroNum += 1;
    }
  }
  return noneZeroNum;
}

template <typename TA, typename TB>
void checkMat2(TA* A, TB* B, int size, string mark) {
  TA* matA = A;
  TB* matB = (TB*)malloc(sizeof(TB) * size);
  int not_passed = 0;
  cudaMemcpy(matB, B, sizeof(TB) * size, cudaMemcpyDeviceToHost);
  float A_nonZero_ratio = float(checkNonZero2(A, size)) / float(size);
  float B_nonZero_ratio = float(checkNonZero(B, size)) / float(size);
  if (A_nonZero_ratio < 0.1 || B_nonZero_ratio < 0.1)
    printf("[checkMat2 %s] nonZero ratio [%f] [%f]\n", mark.c_str(),
           A_nonZero_ratio, B_nonZero_ratio);
  int idx = 0;
  for (int jjj = 0; jjj < size; jjj++) {
    if (fabs(float(matA[jjj]) - float(matB[jjj])) > 0.00001) {
      not_passed += 1;
      if (idx < 1000) {
        printf("%d %f %f %f\n", idx, float(matA[jjj]), float(matB[jjj]),
               float(matA[jjj]) - float(matB[jjj]));
        idx += 1;
      }
    }
  }
  if (not_passed != 0)
    printf("[checkMat2 %s] different elements : %d \n", mark.c_str(),
           not_passed);
  free(matB);
}

template <typename TA, typename TB>
void checkMat3(TA* A, TB* B, int size, float alpha, string mark) {
  TA* matA = A;
  TB* matB = (TB*)malloc(sizeof(TB) * size);
  int not_passed = 0;
  cudaMemcpy(matB, B, sizeof(TB) * size, cudaMemcpyDeviceToHost);
  float A_nonZero_ratio = float(checkNonZero2(A, size)) / float(size);
  float B_nonZero_ratio = float(checkNonZero(B, size)) / float(size);
  if (A_nonZero_ratio < 0.1 || B_nonZero_ratio < 0.1)
    printf("[checkMat3 %s] nonZero ratio [%f] [%f]\n", mark.c_str(),
           A_nonZero_ratio, B_nonZero_ratio);
  int idx = 0;
  for (int jjj = 0; jjj < size; jjj++) {
    if (fabs(float(round(float(matA[jjj]) * alpha)) - float(matB[jjj])) >
        0.00001) {
      not_passed += 1;
      if (idx < 1000) {
        printf("%d %f %f %f\n", idx, float(matA[jjj]) * alpha, float(matB[jjj]),
               float(matA[jjj]) * alpha - float(matB[jjj]));
        idx += 1;
      }
    }
  }
  if (not_passed != 0)
    printf("[checkMat3 %s] different elements : %d \n", mark.c_str(),
           not_passed);
  free(matB);
}
template <typename T>
float getAMax(T* A, int size) {
  T* h_A = (T*)malloc(sizeof(T) * size);
  cudaMemcpy(h_A, A, sizeof(T) * size, cudaMemcpyDeviceToHost);
  float amax = -1000;
  for (int i = 0; i < size; i++)
    if (fabs(float(h_A[i])) > amax) amax = fabs(float(h_A[i]));
  free(h_A);
  return amax;
}

typedef struct {
  int algoId, customOption, tile, splitK_val, swizzle, reductionScheme,
      workspaceSize, stages;
} cublasLtMatmulAlgo_info;

/* Structure to store information about different run trials */
typedef struct {
  cublasLtMatmulAlgo_t algo;
  cublasStatus_t status;
  float time;
  size_t workspaceSize;  // actual memory workspace needed
  cublasMath_t mathMode;
  cublasLtReductionScheme_t reductionScheme;
  int customOption;
  float wavesCount;
} customMatmulPerf_t;

/* CAUTION : must match cublasLtMatmulTile_t */
const char* const matmulTileName[] = {
    "UNDEF",  "8x8",    "8x16",    "16x8",    "8x32",   "16x16",  "32x8",
    "8x64",   "16x32",  "32x16",   "64x8",    "32x32",  "32x64",  "64x32",
    "32x128", "64x64",  "128x32",  "64x128",  "128x64", "64x256", "128x128",
    "256x64", "64x512", "128x256", "256x128", "512x64",
};
// Utility function to print customMatmulPerf_t structure
void printPerfStructure(int m, int n, int k, const customMatmulPerf_t& perf,
                        cublasLtMatmulAlgo_info& best_algo, int algoI) {
  int algoId, tile, swizzle, customOption, numSplitsK, reductionScheme, stages;

  const cublasLtMatmulAlgo_t* matmulAlgo = &perf.algo;
  cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_ID,
                                       &algoId, sizeof(algoId), NULL);
  cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo, CUBLASLT_ALGO_CONFIG_TILE_ID,
                                       &tile, sizeof(tile), NULL);
  cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo,
                                       CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                       &numSplitsK, sizeof(numSplitsK), NULL);
  cublasLtMatmulAlgoConfigGetAttribute(
      matmulAlgo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionScheme,
      sizeof(reductionScheme), NULL);
  cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo,
                                       CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
                                       &swizzle, sizeof(swizzle), NULL);
  cublasLtMatmulAlgoConfigGetAttribute(
      matmulAlgo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption,
      sizeof(customOption), NULL);
  cublasLtMatmulAlgoConfigGetAttribute(matmulAlgo,
                                       CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages,
                                       sizeof(stages), NULL);

  memset(tmp_cstr, 0, sizeof tmp_cstr);
  sprintf(
      tmp_cstr,
      "algo={ Id=%d, tileIdx=%d (%s) splitK=%d reduc=%d swizzle=%d custom=%d "
      "stages=%d} status %d "
      "time %fms workspace=%d mathMode=%d waves=%f\n",
      algoId, tile, matmulTileName[tile], numSplitsK, reductionScheme, swizzle,
      customOption, stages, perf.status, perf.time, (int)perf.workspaceSize,
      (int)perf.mathMode, perf.wavesCount);
  tmp_str = string(tmp_cstr);
  gemm_test_result += tmp_str;
  if (algoI == 0) {
    best_algo.algoId = algoId;
    best_algo.customOption = customOption;
    best_algo.tile = tile;
    best_algo.splitK_val = numSplitsK;
    best_algo.swizzle = swizzle;
    best_algo.reductionScheme = reductionScheme;
    best_algo.stages = stages;
    best_algo.workspaceSize = (int)perf.workspaceSize;
  }
}

static inline bool time_compare(const customMatmulPerf_t& perf_a,
                                const customMatmulPerf_t& perf_b) {
  return ((perf_a.status == CUBLAS_STATUS_SUCCESS) &&
          (perf_a.time < perf_b.time));
}

static cublasStatus_t customMatmulRun(
    cublasLtHandle_t ltHandle,  // to get the capabilities (required a GPU)
    cublasLtMatmulDesc_t operationDesc,
    const void* alpha, /* host or device pointer */
    const void* A, cublasLtMatrixLayout_t Adesc, const void* B,
    cublasLtMatrixLayout_t Bdesc, const void* beta, /* host or device pointer */
    const void* C, cublasLtMatrixLayout_t Cdesc, void* D,
    cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t& algo,
    int kernelRepeats, void* workSpace, size_t workSpaceSizeInBytes,
    customMatmulPerf_t& perfResults, cudaStream_t stream,
    cudaEvent_t& startEvent, cudaEvent_t& stopEvent) {
  cublasLtMatmulHeuristicResult_t heurResult;
  /* Looping over the Algo */
  int repeats = kernelRepeats;
  cublasStatus_t algoStatus = cublasLtMatmulAlgoCheck(
      ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, &algo, &heurResult);
  // printf("cublasLtMatmulAlgoCheck status %d\n", algoStatus);
  if (algoStatus == CUBLAS_STATUS_SUCCESS) {
    if (heurResult.workspaceSize <= workSpaceSizeInBytes) {
      cublasStatus_t oneRunStatus;
      float time;

      oneRunStatus = cublasLtMatmul(ltHandle, operationDesc, alpha, A, Adesc, B,
                                    Bdesc, beta, C, Cdesc, D, Ddesc, &algo,
                                    workSpace, workSpaceSizeInBytes, stream);
      cudaEventRecord(startEvent, stream);
      for (int loop = 1; loop < repeats; loop++) {
        oneRunStatus = cublasLtMatmul(ltHandle, operationDesc, alpha, A, Adesc,
                                      B, Bdesc, beta, C, Cdesc, D, Ddesc, &algo,
                                      workSpace, workSpaceSizeInBytes, stream);
      }
      cudaEventRecord(stopEvent, stream);
      cudaEventSynchronize(startEvent);
      cudaEventSynchronize(stopEvent);
      cudaEventElapsedTime(&time, startEvent, stopEvent);
      time = time / (repeats - 1);
      algoStatus = oneRunStatus;
      if (algoStatus == CUBLAS_STATUS_SUCCESS) {
        perfResults.algo = algo;
        perfResults.time = time;
        perfResults.workspaceSize = heurResult.workspaceSize;
        perfResults.wavesCount = heurResult.wavesCount;
      }
    } else {
      printf("not enough workspace! %ld\n", heurResult.workspaceSize);
      algoStatus = CUBLAS_STATUS_NOT_SUPPORTED;  // Not enough workspace
    }
  } else {
  }

  return algoStatus;
}

// find the best algo for INT8 gemm + cublasLt + INT8 Output
int LtIgemmCustomFindINT8OutputCOL32_2R_4R4(
    cublasLtHandle_t ltHandle, int m, int n, int k,
    const float* alpha,                                  /* host pointer */
    const int8_t* A, const int8_t* B, const float* beta, /* host pointer */
    int8_t* C, cublasLtMatmulAlgo_info& best_algo) {
  void* workSpace = NULL;
  size_t workSpaceSize = sizeof(char*) * 64 * m * n;
  cudaMalloc((void**)&workSpace, workSpaceSize);
  cudaEvent_t startEvent;
  cudaEvent_t stopEvent;
  cudaEventCreate(&startEvent, cudaEventBlockingSync);
  cudaEventCreate(&stopEvent, cudaEventBlockingSync);

  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  cublasLtMatmulDesc_t operationDesc = NULL;
  cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  float transformAlpha = 1.0f, transformBeta = 0.0f;
  int8_t *Atransform = NULL, *Btransform = NULL;
  int32_t* Ctransform = NULL;
  cublasLtMatrixLayout_t AtransformDesc = NULL, BtransformDesc = NULL,
                         CtransformDesc = NULL;
  cudaStream_t stream = 0;
  // SplitK value that we are going to try when SplitK is supported for a given
  // algo
  const int splitKSequenceA[] = {2, 3, 4, 5, 6, 8, 12, 16, 32};
// Let try a fixed number of combinations
#define ALGO_COMBINATIONS 50000
  int AlgoCombinations = ALGO_COMBINATIONS;
  int AlgoCount = 0;
  int kernelRepeats =
      10;  // number of time the CUDA kernels will be run back to back
  customMatmulPerf_t perfResults[ALGO_COMBINATIONS];
  int nbAlgoIds = 0;
#define ALGO_IDS 100
  int algoIdA[ALGO_IDS];
  cudaDataType_t scaleType = CUDA_R_32F, Atype = CUDA_R_8I, Btype = CUDA_R_8I,
                 Ctype = CUDA_R_8I;
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
  cublasOperation_t opTranspose = CUBLAS_OP_T;
  cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;
  cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;
  cublasLtOrder_t order_COL32_2R_4R4 = CUBLASLT_ORDER_COL32_2R_4R4;
  cublasLtMatrixTransformDesc_t transformDesc = NULL;
  int ldaTransform = 32 * m;
  int ldbTransform;
  if (RUN_WITH_A100 == 0)
    ldbTransform = 32 * roundoff(n, 8);
  else
    ldbTransform = 32 * roundoff(n, 32);
  int ldcTransform = 32 * m;

  cudaMalloc(&Atransform, sizeof(int8_t) * roundoff(k, 32) / 32 * ldaTransform);
  cudaMalloc(&Btransform, sizeof(int8_t) * roundoff(k, 32) / 32 * ldbTransform);
  cudaMalloc(&Ctransform, sizeof(int8_t) * roundoff(n, 32) / 32 * ldcTransform);

  cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_32F);

  // --------------------------------------
  // Create descriptors for the original matrices
  cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, m, k, m);
  cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, n, k, n);
  cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_8I, m, n, m);

  // -----------------------------------------------------------
  // Create descriptors for the transformed matrices
  cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, m, k, ldaTransform);
  cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_COL32, sizeof(order_COL32));

  cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, n, k, ldbTransform);
  if (RUN_WITH_A100 == 0)
    cublasLtMatrixLayoutSetAttribute(
        BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL4_4R2_8C,
        sizeof(order_COL4_4R2_8C));
  else
    cublasLtMatrixLayoutSetAttribute(
        BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32_2R_4R4,
        sizeof(order_COL32_2R_4R4));

  cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_8I, m, n, ldcTransform);
  cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_COL32, sizeof(order_COL32));

  // --------------------------------------------------------
  // Transforms
  cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, A, Adesc,
                          &transformBeta, NULL, NULL, Atransform,
                          AtransformDesc, 0);
  cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, B, Bdesc,
                          &transformBeta, NULL, NULL, Btransform,
                          BtransformDesc, 0);

  cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType);
  // Tensor op igemm kernels only support NT gemm
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                 &opTranspose, sizeof(cublasOperation_t));
  // using alpha to quantize
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE,
                                 &scaleType, sizeof(scaleType));

  // Request the AlgoId available
  cublasLtMatmulAlgoGetIds(ltHandle, computeType, scaleType, Atype, Btype,
                           Ctype, Ctype, ALGO_IDS, algoIdA, &nbAlgoIds);

  // Loop over the Algo IDs
  for (int idx = 0; (idx < nbAlgoIds) && (AlgoCount < AlgoCombinations);
       idx++) {
    cublasLtMatmulAlgo_t algo;
    size_t sizeWritten = 0;
    /* Initialize algo structure with given Algp ID */
    status = cublasLtMatmulAlgoInit(ltHandle, computeType, scaleType, Atype,
                                    Btype, Ctype, Ctype, algoIdA[idx], &algo);
    if (status != CUBLAS_STATUS_SUCCESS) {
      continue;
    }
    // Query the tiles enums supported by that algo
    cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, NULL,
                                      0, &sizeWritten);
    int nbTiles = int(sizeWritten / sizeof(int));
    int* tileA = new int[nbTiles == 0 ? 1 : nbTiles];
    if (nbTiles == 0) {
      tileA[0] = CUBLASLT_MATMUL_TILE_UNDEFINED;
      nbTiles = 1;
    }
    cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, NULL,
                                      0, &sizeWritten);
    int nbStages = int(sizeWritten / sizeof(int));
    std::vector<int> stagesA(nbStages == 0 ? 1 : nbStages);
    if (nbStages == 0) {
      stagesA[0] = CUBLASLT_MATMUL_STAGES_UNDEFINED;
      nbStages = 1;
    } else {
      cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS,
                                        stagesA.data(), sizeof(int) * nbStages,
                                        &sizeWritten);
    }
    int splitkSupport, redMask, swizzlingMax, customOptionMax;
    // Retrieve Algo Capabilities attributes to be able to setup loop over the
    // different combinations
    cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, tileA,
                                      sizeof(int) * nbTiles, &sizeWritten);
    cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT,
                                      &splitkSupport, sizeof(splitkSupport),
                                      &sizeWritten);
    cublasLtMatmulAlgoCapGetAttribute(&algo,
                                      CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK,
                                      &redMask, sizeof(redMask), &sizeWritten);
    cublasLtMatmulAlgoCapGetAttribute(
        &algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swizzlingMax,
        sizeof(swizzlingMax), &sizeWritten);
    cublasLtMatmulAlgoCapGetAttribute(
        &algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &customOptionMax,
        sizeof(customOptionMax), &sizeWritten);
    /* Loop over the different tiles */
    for (int tileIdx = 0; tileIdx < nbTiles; tileIdx++) {
      /* Loop over different stages count */
      for (int stagesIdx = 0; stagesIdx < nbStages; stagesIdx++) {
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stagesA[stagesIdx],
            sizeof(stagesA[stagesIdx]));
        /* Loop over the different custom option if any */
        for (int customOption = 0; customOption <= customOptionMax;
             customOption++) {
          cublasLtMatmulAlgoConfigSetAttribute(
              &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption,
              sizeof(customOption));
          /* Loop over the CTAs swizzling support */
          for (int k = 0; k <= swizzlingMax; k++) {
            int splitK_trial = 0;
            if (splitkSupport) {
              splitK_trial +=
                  sizeof(splitKSequenceA) / sizeof(splitKSequenceA[0]);
            }
            // Loop over the splitK value over a fixed sequence splitKSequenceA
            // in addtion to the case where splitK is not enabled
            for (int l = 0;
                 (l < (1 + splitK_trial)) && (AlgoCount < AlgoCombinations);
                 l++) {
              /* Setup attribute of the algo to run */
              cublasLtMatmulAlgoConfigSetAttribute(
                  &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileA[tileIdx],
                  sizeof(tileA[tileIdx]));
              int splitK_val = 0;
              int redScheme = CUBLASLT_REDUCTION_SCHEME_NONE;
              cublasLtMatmulAlgoConfigSetAttribute(
                  &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK_val,
                  sizeof(splitK_val));
              cublasLtMatmulAlgoConfigSetAttribute(
                  &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &k, sizeof(k));
              cublasLtMatmulAlgoConfigSetAttribute(
                  &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme,
                  sizeof(int));

              if (l > 0) {  // Split-K case
                splitK_val = splitKSequenceA[l - 1];
                cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                    &splitKSequenceA[l - 1], sizeof(splitKSequenceA[l - 1]));
                /* Going over all the reduction scheme  */
                for (redScheme = 1;
                     redScheme <= (int)CUBLASLT_REDUCTION_SCHEME_MASK &&
                     (AlgoCount < AlgoCombinations);
                     redScheme = redScheme << 1) {
                  if (redScheme & redMask) {
                    cublasLtMatmulAlgoConfigSetAttribute(
                        &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                        &redScheme, sizeof(redScheme));
                    status = customMatmulRun(
                        ltHandle, operationDesc,
                        alpha, /* host or device pointer */
                        Atransform, AtransformDesc, Btransform, BtransformDesc,
                        beta, /* host or device pointer */
                        Ctransform, CtransformDesc, Ctransform, CtransformDesc,
                        algo, kernelRepeats, workSpace, workSpaceSize,
                        perfResults[AlgoCount], stream, startEvent, stopEvent);
                    perfResults[AlgoCount].status = status;
                    if (status == CUBLAS_STATUS_SUCCESS) AlgoCount++;
                  }     // end if
                }       // end for
              } else {  // Non-splitK case
                /* if user preference is ok with workspace */
                if (AlgoCount < AlgoCombinations) {
                  status = customMatmulRun(
                      ltHandle, operationDesc,
                      alpha, /* host or device pointer */
                      Atransform, AtransformDesc, Btransform, BtransformDesc,
                      beta, /* host or device pointer */
                      Ctransform, CtransformDesc, Ctransform, CtransformDesc,
                      algo, kernelRepeats, workSpace, workSpaceSize,
                      perfResults[AlgoCount], stream, startEvent, stopEvent);
                  perfResults[AlgoCount].status = status;
                  if (status == CUBLAS_STATUS_SUCCESS) AlgoCount++;
                }
              }
            }  // end l
          }    // end k
        }      // end customOption
      }
    }  // end tileIdx
    delete[] tileA;
  }  // end idx
  // Sort the results per run duration
  std::sort(perfResults, perfResults + AlgoCount, time_compare);
  // Print timing and perf details
  for (int i = 0; i < AlgoCount; i++) {
    memset(tmp_cstr, 0, sizeof tmp_cstr);
    sprintf(tmp_cstr, "INT8 NT-gemm %s INT8 IO cublasLt %03d : ",
            RUN_WITH_A100 ? "CUBLASLT_ORDER_COL32_2R_4R4"
                          : "CUBLASLT_ORDER_COL4_4R2_8C",
            i);
    tmp_str = string(tmp_cstr);
    gemm_test_result += tmp_str;
    printPerfStructure(m, n, k, perfResults[i], best_algo, i);
    break;
  }

  // Descriptors are no longer needed as all GPU work was already enqueued
  if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
  if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
  if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
  if (CtransformDesc) cublasLtMatrixLayoutDestroy(CtransformDesc);
  if (BtransformDesc) cublasLtMatrixLayoutDestroy(BtransformDesc);
  if (AtransformDesc) cublasLtMatrixLayoutDestroy(AtransformDesc);
  if (operationDesc) cublasLtMatmulDescDestroy(operationDesc);
  if (transformDesc) cublasLtMatrixTransformDescDestroy(transformDesc);
  if (Atransform) cudaFree(Atransform);
  if (Btransform) cudaFree(Btransform);
  if (Ctransform) cudaFree(Ctransform);
  if (workSpace) cudaFree(workSpace);
  if (startEvent) cudaEventDestroy(startEvent);
  if (stopEvent) cudaEventDestroy(stopEvent);

  return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}

void int8gemm_with_cublasLtMatmul_ORDER_COL32_2R_4R4(
    int m, int n, int k, float alpha2, float beta2, int8_t* d_int8_A,
    int8_t* d_int8_B, int8_t* d_int8_C,
    cublasLtMatmulAlgo_info& INT8_gemm_cublasLt_INT8Output_best_algo,
    cublasLtHandle_t ltHandle, cudaStream_t stream) {
  cublasStatus_t cublasStat;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int iters = 10;
  cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL;
  float transformAlpha = 1.0f, transformBeta = 0.0f;
  int8_t *Atransform = NULL, *Btransform = NULL;
  cublasLtMatrixLayout_t AtransformDesc = NULL, BtransformDesc = NULL,
                         CtransformDesc = NULL;
  cublasOperation_t opTranspose = CUBLAS_OP_T;
  cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;
  cublasLtOrder_t order_COL4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C;
  cublasLtOrder_t order_COL32_2R_4R4 = CUBLASLT_ORDER_COL32_2R_4R4;
  cublasLtMatrixTransformDesc_t transformDesc = NULL;
  int ldaTransform = 32 * m;
  int ldbTransform;
  if (RUN_WITH_A100 == 0)
    ldbTransform = 32 * roundoff(n, 8);
  else
    ldbTransform = 32 * roundoff(n, 32);
  int ldcTransform = 32 * m;

  cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_32F);

  int8_t* Ctransform2 = NULL;
  cudaDataType_t scaleType, Atype, Btype, Ctype;
  cublasComputeType_t computeType;
  computeType = CUBLAS_COMPUTE_32I, scaleType = CUDA_R_32F, Atype = CUDA_R_8I,
  Btype = CUDA_R_8I, Ctype = CUDA_R_8I;

  cublasLtMatmulDesc_t operationDesc2 = NULL;
  cublasLtMatrixLayout_t Cdesc2 = NULL;
  cublasLtMatrixLayout_t CtransformDesc2 = NULL;

  cudaMalloc(&Atransform, sizeof(int8_t) * roundoff(k, 32) / 32 * ldaTransform);
  cudaMalloc(&Btransform, sizeof(int8_t) * roundoff(k, 32) / 32 * ldbTransform);
  cudaMalloc(&Ctransform2,
             sizeof(int8_t) * roundoff(n, 32) / 32 * ldcTransform);

  // --------------------------------------
  // Create descriptors for the original matrices
  cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, m, k, m);
  cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, n, k, n);
  cublasLtMatrixLayoutCreate(&Cdesc2, CUDA_R_8I, m, n, m);

  // -----------------------------------------------------------
  // Create descriptors for the transformed matrices
  cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, m, k, ldaTransform);
  cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_COL32, sizeof(order_COL32));

  cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, n, k, ldbTransform);
  if (RUN_WITH_A100 == 0)
    cublasLtMatrixLayoutSetAttribute(
        BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL4_4R2_8C,
        sizeof(order_COL4_4R2_8C));
  else
    cublasLtMatrixLayoutSetAttribute(
        BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32_2R_4R4,
        sizeof(order_COL32_2R_4R4));
  cublasLtMatrixLayoutCreate(&CtransformDesc2, CUDA_R_8I, m, n, ldcTransform);
  cublasLtMatrixLayoutSetAttribute(CtransformDesc2,
                                   CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32,
                                   sizeof(order_COL32));

  // --------------------------------------------------------

  cublasLtMatmulDescCreate(&operationDesc2, computeType, scaleType);
  // Tensor op igemm kernels only support NT gemm
  cublasLtMatmulDescSetAttribute(operationDesc2, CUBLASLT_MATMUL_DESC_TRANSB,
                                 &opTranspose, sizeof(cublasOperation_t));
  // using alpha to quantize
  cublasLtMatmulDescSetAttribute(operationDesc2,
                                 CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scaleType,
                                 sizeof(scaleType));

  // get algo
  cublasLtMatmulAlgo_t algo2;
  char* workSpace2 = NULL;
  int workspaceSize = 0;
  workspaceSize = INT8_gemm_cublasLt_INT8Output_best_algo.workspaceSize;
  cublasLtMatmulAlgoInit(
      ltHandle, computeType, CUDA_R_32F, CUDA_R_8I, CUDA_R_8I, CUDA_R_8I,
      CUDA_R_8I, INT8_gemm_cublasLt_INT8Output_best_algo.algoId, &algo2);
  cublasLtMatmulAlgoConfigSetAttribute(
      &algo2, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
      &(INT8_gemm_cublasLt_INT8Output_best_algo.customOption),
      sizeof(INT8_gemm_cublasLt_INT8Output_best_algo.customOption));
  cublasLtMatmulAlgoConfigSetAttribute(
      &algo2, CUBLASLT_ALGO_CONFIG_TILE_ID,
      &(INT8_gemm_cublasLt_INT8Output_best_algo.tile),
      sizeof(INT8_gemm_cublasLt_INT8Output_best_algo.tile));
  cublasLtMatmulAlgoConfigSetAttribute(
      &algo2, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
      &(INT8_gemm_cublasLt_INT8Output_best_algo.splitK_val),
      sizeof(INT8_gemm_cublasLt_INT8Output_best_algo.splitK_val));
  cublasLtMatmulAlgoConfigSetAttribute(
      &algo2, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
      &(INT8_gemm_cublasLt_INT8Output_best_algo.swizzle),
      sizeof(INT8_gemm_cublasLt_INT8Output_best_algo.swizzle));
  cublasLtMatmulAlgoConfigSetAttribute(
      &algo2, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
      &(INT8_gemm_cublasLt_INT8Output_best_algo.reductionScheme), sizeof(int));
  cublasLtMatmulAlgoConfigSetAttribute(
      &algo2, CUBLASLT_ALGO_CONFIG_STAGES_ID,
      &(INT8_gemm_cublasLt_INT8Output_best_algo.stages),
      sizeof(INT8_gemm_cublasLt_INT8Output_best_algo.stages));
  if (INT8_gemm_cublasLt_INT8Output_best_algo.workspaceSize != 0) {
    cudaMalloc(
        (void**)&workSpace2,
        sizeof(char) * INT8_gemm_cublasLt_INT8Output_best_algo.workspaceSize);
  }

  cublasStat = cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha,
                                       d_int8_A, Adesc, &transformBeta, NULL,
                                       NULL, Atransform, AtransformDesc, 0);
  cublasStat = cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha,
                                       d_int8_B, Bdesc, &transformBeta, NULL,
                                       NULL, Btransform, BtransformDesc, 0);

  float time_used;
  cublasStat = cublasLtMatmul(
      ltHandle, operationDesc2, &alpha2, Atransform, AtransformDesc, Btransform,
      BtransformDesc, &beta2, Ctransform2, CtransformDesc2, Ctransform2,
      CtransformDesc2, &algo2, workSpace2, workspaceSize, stream);
  cudaEventRecord(start, 0);
  for (int t = 1; t < iters; t++) {
    cublasStat = cublasLtMatmul(ltHandle, operationDesc2, &alpha2, Atransform,
                                AtransformDesc, Btransform, BtransformDesc,
                                &beta2, Ctransform2, CtransformDesc2,
                                Ctransform2, CtransformDesc2, &algo2,
                                workSpace2, workspaceSize, stream);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(start);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_used, start, stop);
  time_used /= (iters - 1);
  if (cublasStat == CUBLAS_STATUS_SUCCESS) {
    memset(tmp_cstr, 0, sizeof tmp_cstr);
    sprintf(tmp_cstr,
            "INT8 NT-gemm with B = %s cublasLtMatmul INT8 output "
            "best algo %d exec_time %f(ms)\n",
            RUN_WITH_A100 ? "CUBLASLT_ORDER_COL32_2R_4R4"
                          : "CUBLASLT_ORDER_COL4_4R2_8C",
            INT8_gemm_cublasLt_INT8Output_best_algo.algoId, time_used);
    tmp_str = string(tmp_cstr);
    gemm_test_result += tmp_str;
  } else {
    std::cout << cublasStat << std::endl;
  }

  cublasStat = cublasLtMatrixTransform(
      ltHandle, transformDesc, &transformAlpha, Ctransform2, CtransformDesc2,
      &transformBeta, NULL, NULL, d_int8_C, Cdesc2, stream);

  if (Cdesc2) cublasLtMatrixLayoutDestroy(Cdesc2);
  if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
  if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
  if (CtransformDesc2) cublasLtMatrixLayoutDestroy(CtransformDesc2);
  if (BtransformDesc) cublasLtMatrixLayoutDestroy(BtransformDesc);
  if (AtransformDesc) cublasLtMatrixLayoutDestroy(AtransformDesc);
  if (operationDesc2) cublasLtMatmulDescDestroy(operationDesc2);
  if (transformDesc) cublasLtMatrixTransformDescDestroy(transformDesc);
  if (Atransform) cudaFree(Atransform);
  if (Btransform) cudaFree(Btransform);
  if (Ctransform2) cudaFree(Ctransform2);
  if (workSpace2) cudaFree(workSpace2);

  checkCudaAPIErrors(cudaEventDestroy(start));
  checkCudaAPIErrors(cudaEventDestroy(stop));
}

// find the best algo for INT8 gemm + cublasLt + INT8 Output
int LtIgemmCustomFindINT8OutputColMajor(cublasLtHandle_t ltHandle, int m, int n,
                                        int k,
                                        const float* alpha, /* host pointer */
                                        const int8_t* A,    // k * m col-major
                                        const int8_t* B,    // k * n col-major
                                        const float* beta,  /* host pointer */
                                        int8_t* C,          // m * n col-major
                                        cublasLtMatmulAlgo_info& best_algo) {
  void* workSpace = NULL;
  size_t workSpaceSize = sizeof(char*) * 64 * m * n;
  cudaMalloc((void**)&workSpace, workSpaceSize);
  cudaEvent_t startEvent;
  cudaEvent_t stopEvent;
  cudaEventCreate(&startEvent, cudaEventBlockingSync);
  cudaEventCreate(&stopEvent, cudaEventBlockingSync);

  cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

  cublasLtMatmulDesc_t operationDesc = NULL;
  cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  float transformAlpha = 1.0f, transformBeta = 0.0f;
  int8_t *Atransform = NULL, *Btransform = NULL;
  int32_t* Ctransform = NULL;
  cublasLtMatrixLayout_t AtransformDesc = NULL, BtransformDesc = NULL,
                         CtransformDesc = NULL;
  cudaStream_t stream = 0;
  // SplitK value that we are going to try when SplitK is supported for a given
  // algo
  const int splitKSequenceA[] = {2, 3, 4, 5, 6, 8, 12, 16, 32};
// Let try a fixed number of combinations
#define ALGO_COMBINATIONS 50000
  int AlgoCombinations = ALGO_COMBINATIONS;
  int AlgoCount = 0;
  int kernelRepeats =
      10;  // number of time the CUDA kernels will be run back to back
  customMatmulPerf_t perfResults[ALGO_COMBINATIONS];
  int nbAlgoIds = 0;
#define ALGO_IDS 100
  int algoIdA[ALGO_IDS];
  cudaDataType_t scaleType = CUDA_R_32F, Atype = CUDA_R_8I, Btype = CUDA_R_8I,
                 Ctype = CUDA_R_8I;
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
  cublasOperation_t opTranspose = CUBLAS_OP_T;
  cublasLtOrder_t order_col_major = CUBLASLT_ORDER_COL;
  cublasLtMatrixTransformDesc_t transformDesc = NULL;

  cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_32F);

  // --------------------------------------
  // Create descriptors for the original matrices
  cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, k, m, k);
  cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_col_major, sizeof(order_col_major));
  cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, k, n, k);
  cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_col_major, sizeof(order_col_major));
  cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_8I, m, n, m);
  cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_col_major, sizeof(order_col_major));

  cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType);
  // Tensor op igemm kernels only support TN gemm
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                 &opTranspose, sizeof(cublasOperation_t));
  // using alpha to quantize
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE,
                                 &scaleType, sizeof(scaleType));

  // Request the AlgoId available
  cublasLtMatmulAlgoGetIds(ltHandle, computeType, scaleType, Atype, Btype,
                           Ctype, Ctype, ALGO_IDS, algoIdA, &nbAlgoIds);

  // Loop over the Algo IDs
  for (int idx = 0; (idx < nbAlgoIds) && (AlgoCount < AlgoCombinations);
       idx++) {
    cublasLtMatmulAlgo_t algo;
    size_t sizeWritten = 0;
    /* Initialize algo structure with given Algp ID */
    status = cublasLtMatmulAlgoInit(ltHandle, computeType, scaleType, Atype,
                                    Btype, Ctype, Ctype, algoIdA[idx], &algo);
    if (status != CUBLAS_STATUS_SUCCESS) {
      continue;
    }
    // Query the tiles enums supported by that algo
    cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, NULL,
                                      0, &sizeWritten);
    int nbTiles = int(sizeWritten / sizeof(int));
    int* tileA = new int[nbTiles == 0 ? 1 : nbTiles];
    if (nbTiles == 0) {
      tileA[0] = CUBLASLT_MATMUL_TILE_UNDEFINED;
      nbTiles = 1;
    }
    cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, NULL,
                                      0, &sizeWritten);
    int nbStages = int(sizeWritten / sizeof(int));
    std::vector<int> stagesA(nbStages == 0 ? 1 : nbStages);
    if (nbStages == 0) {
      stagesA[0] = CUBLASLT_MATMUL_STAGES_UNDEFINED;
      nbStages = 1;
    } else {
      cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS,
                                        stagesA.data(), sizeof(int) * nbStages,
                                        &sizeWritten);
    }
    int splitkSupport, redMask, swizzlingMax, customOptionMax;
    // Retrieve Algo Capabilities attributes to be able to setup loop over the
    // different combinations
    cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, tileA,
                                      sizeof(int) * nbTiles, &sizeWritten);
    cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT,
                                      &splitkSupport, sizeof(splitkSupport),
                                      &sizeWritten);
    cublasLtMatmulAlgoCapGetAttribute(&algo,
                                      CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK,
                                      &redMask, sizeof(redMask), &sizeWritten);
    cublasLtMatmulAlgoCapGetAttribute(
        &algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swizzlingMax,
        sizeof(swizzlingMax), &sizeWritten);
    cublasLtMatmulAlgoCapGetAttribute(
        &algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &customOptionMax,
        sizeof(customOptionMax), &sizeWritten);
    /* Loop over the different tiles */
    for (int tileIdx = 0; tileIdx < nbTiles; tileIdx++) {
      /* Loop over different stages count */
      for (int stagesIdx = 0; stagesIdx < nbStages; stagesIdx++) {
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stagesA[stagesIdx],
            sizeof(stagesA[stagesIdx]));
        /* Loop over the different custom option if any */
        for (int customOption = 0; customOption <= customOptionMax;
             customOption++) {
          cublasLtMatmulAlgoConfigSetAttribute(
              &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption,
              sizeof(customOption));
          /* Loop over the CTAs swizzling support */
          for (int k = 0; k <= swizzlingMax; k++) {
            int splitK_trial = 0;
            if (splitkSupport) {
              splitK_trial +=
                  sizeof(splitKSequenceA) / sizeof(splitKSequenceA[0]);
            }
            // Loop over the splitK value over a fixed sequence splitKSequenceA
            // in addtion to the case where splitK is not enabled
            for (int l = 0;
                 (l < (1 + splitK_trial)) && (AlgoCount < AlgoCombinations);
                 l++) {
              /* Setup attribute of the algo to run */
              cublasLtMatmulAlgoConfigSetAttribute(
                  &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileA[tileIdx],
                  sizeof(tileA[tileIdx]));
              int splitK_val = 0;
              int redScheme = CUBLASLT_REDUCTION_SCHEME_NONE;
              cublasLtMatmulAlgoConfigSetAttribute(
                  &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK_val,
                  sizeof(splitK_val));
              cublasLtMatmulAlgoConfigSetAttribute(
                  &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &k, sizeof(k));
              cublasLtMatmulAlgoConfigSetAttribute(
                  &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme,
                  sizeof(int));

              if (l > 0) {  // Split-K case
                splitK_val = splitKSequenceA[l - 1];
                cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                    &splitKSequenceA[l - 1], sizeof(splitKSequenceA[l - 1]));
                /* Going over all the reduction scheme  */
                for (redScheme = 1;
                     redScheme <= (int)CUBLASLT_REDUCTION_SCHEME_MASK &&
                     (AlgoCount < AlgoCombinations);
                     redScheme = redScheme << 1) {
                  if (redScheme & redMask) {
                    cublasLtMatmulAlgoConfigSetAttribute(
                        &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                        &redScheme, sizeof(redScheme));
                    status = customMatmulRun(
                        ltHandle, operationDesc,
                        alpha,                    /* host or device pointer */
                        A, Adesc, B, Bdesc, beta, /* host or device pointer */
                        C, Cdesc, C, Cdesc, algo, kernelRepeats, workSpace,
                        workSpaceSize, perfResults[AlgoCount], stream,
                        startEvent, stopEvent);
                    perfResults[AlgoCount].status = status;
                    if (status == CUBLAS_STATUS_SUCCESS) AlgoCount++;
                  }     // end if
                }       // end for
              } else {  // Non-splitK case
                /* if user preference is ok with workspace */
                if (AlgoCount < AlgoCombinations) {
                  status = customMatmulRun(
                      ltHandle, operationDesc,
                      alpha,                    /* host or device pointer */
                      A, Adesc, B, Bdesc, beta, /* host or device pointer */
                      C, Cdesc, C, Cdesc, algo, kernelRepeats, workSpace,
                      workSpaceSize, perfResults[AlgoCount], stream, startEvent,
                      stopEvent);
                  perfResults[AlgoCount].status = status;
                  if (status == CUBLAS_STATUS_SUCCESS) AlgoCount++;
                }
              }
            }  // end l
          }    // end k
        }      // end customOption
      }
    }  // end tileIdx
    delete[] tileA;
  }  // end idx
  // Sort the results per run duration
  std::sort(perfResults, perfResults + AlgoCount, time_compare);
  // Print timing and perf details
  for (int i = 0; i < AlgoCount; i++) {
    memset(tmp_cstr, 0, sizeof tmp_cstr);
    sprintf(tmp_cstr,
            "INT8 TN-gemm CUBLASLT_ORDER_COL INT8 IO cublasLt %03d : ", i);
    tmp_str = string(tmp_cstr);
    gemm_test_result += tmp_str;
    printPerfStructure(m, n, k, perfResults[i], best_algo, i);
    break;
  }

  // Descriptors are no longer needed as all GPU work was already enqueued
  if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
  if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
  if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
  if (operationDesc) cublasLtMatmulDescDestroy(operationDesc);
  if (workSpace) cudaFree(workSpace);
  if (startEvent) cudaEventDestroy(startEvent);
  if (stopEvent) cudaEventDestroy(stopEvent);

  return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}

void int8gemm_with_cublasLtMatmul_COL_MAJOR(
    int m, int n, int k, float alpha2, float beta2, int8_t* A, int8_t* B,
    int8_t* C, cublasLtMatmulAlgo_info& INT8_gemm_cublasLt_INT8Output_best_algo,
    cublasLtHandle_t ltHandle, cudaStream_t stream) {
  cublasStatus_t cublasStat;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int iters = 10;
  cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL;
  float transformAlpha = 1.0f, transformBeta = 0.0f;
  cublasLtMatrixLayout_t AtransformDesc = NULL, BtransformDesc = NULL,
                         CtransformDesc = NULL;
  cublasOperation_t opTranspose = CUBLAS_OP_T;
  cublasLtOrder_t order_COL_MAJOR = CUBLASLT_ORDER_COL;

  cublasLtMatrixTransformDesc_t transformDesc = NULL;

  cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_32F);

  cudaDataType_t scaleType, Atype, Btype, Ctype;
  cublasComputeType_t computeType;
  computeType = CUBLAS_COMPUTE_32I, scaleType = CUDA_R_32F, Atype = CUDA_R_8I,
  Btype = CUDA_R_8I, Ctype = CUDA_R_8I;

  cublasLtMatmulDesc_t operationDesc2 = NULL;
  cublasLtMatrixLayout_t Cdesc2 = NULL;
  cublasLtMatrixLayout_t CtransformDesc2 = NULL;

  // --------------------------------------
  // Create descriptors for the original matrices
  cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, k, m, k);
  cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_COL_MAJOR, sizeof(order_COL_MAJOR));
  cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, k, n, k);
  cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_COL_MAJOR, sizeof(order_COL_MAJOR));
  cublasLtMatrixLayoutCreate(&Cdesc2, CUDA_R_8I, m, n, m);
  cublasLtMatrixLayoutSetAttribute(Cdesc2, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_COL_MAJOR, sizeof(order_COL_MAJOR));

  cublasLtMatmulDescCreate(&operationDesc2, computeType, scaleType);
  // Tensor op igemm kernels only support NT gemm
  cublasLtMatmulDescSetAttribute(operationDesc2, CUBLASLT_MATMUL_DESC_TRANSA,
                                 &opTranspose, sizeof(cublasOperation_t));
  // using alpha to quantize
  cublasLtMatmulDescSetAttribute(operationDesc2,
                                 CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scaleType,
                                 sizeof(scaleType));

  // get algo
  cublasLtMatmulAlgo_t algo2;
  char* workSpace2 = NULL;
  int workspaceSize = 0;
  workspaceSize = INT8_gemm_cublasLt_INT8Output_best_algo.workspaceSize;
  cublasLtMatmulAlgoInit(
      ltHandle, computeType, CUDA_R_32F, CUDA_R_8I, CUDA_R_8I, CUDA_R_8I,
      CUDA_R_8I, INT8_gemm_cublasLt_INT8Output_best_algo.algoId, &algo2);
  cublasLtMatmulAlgoConfigSetAttribute(
      &algo2, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
      &(INT8_gemm_cublasLt_INT8Output_best_algo.customOption),
      sizeof(INT8_gemm_cublasLt_INT8Output_best_algo.customOption));
  cublasLtMatmulAlgoConfigSetAttribute(
      &algo2, CUBLASLT_ALGO_CONFIG_TILE_ID,
      &(INT8_gemm_cublasLt_INT8Output_best_algo.tile),
      sizeof(INT8_gemm_cublasLt_INT8Output_best_algo.tile));
  cublasLtMatmulAlgoConfigSetAttribute(
      &algo2, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
      &(INT8_gemm_cublasLt_INT8Output_best_algo.splitK_val),
      sizeof(INT8_gemm_cublasLt_INT8Output_best_algo.splitK_val));
  cublasLtMatmulAlgoConfigSetAttribute(
      &algo2, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
      &(INT8_gemm_cublasLt_INT8Output_best_algo.swizzle),
      sizeof(INT8_gemm_cublasLt_INT8Output_best_algo.swizzle));
  cublasLtMatmulAlgoConfigSetAttribute(
      &algo2, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
      &(INT8_gemm_cublasLt_INT8Output_best_algo.reductionScheme), sizeof(int));
  cublasLtMatmulAlgoConfigSetAttribute(
      &algo2, CUBLASLT_ALGO_CONFIG_STAGES_ID,
      &(INT8_gemm_cublasLt_INT8Output_best_algo.stages),
      sizeof(INT8_gemm_cublasLt_INT8Output_best_algo.stages));
  if (INT8_gemm_cublasLt_INT8Output_best_algo.workspaceSize != 0) {
    cudaMalloc(
        (void**)&workSpace2,
        sizeof(char) * INT8_gemm_cublasLt_INT8Output_best_algo.workspaceSize);
  }

  float time_used;
  cublasStat = cublasLtMatmul(ltHandle, operationDesc2, &alpha2, A, Adesc, B,
                              Bdesc, &beta2, C, Cdesc2, C, Cdesc2, &algo2,
                              workSpace2, workspaceSize, stream);
  cudaEventRecord(start, 0);
  for (int t = 1; t < iters; t++) {
    cublasStat = cublasLtMatmul(ltHandle, operationDesc2, &alpha2, A, Adesc, B,
                                Bdesc, &beta2, C, Cdesc2, C, Cdesc2, &algo2,
                                workSpace2, workspaceSize, stream);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(start);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_used, start, stop);
  time_used /= (iters - 1);
  if (cublasStat == CUBLAS_STATUS_SUCCESS) {
    memset(tmp_cstr, 0, sizeof tmp_cstr);
    sprintf(
        tmp_cstr,
        "INT8 TN-gemm with B = CUBLASLT_ORDER_COL cublasLtMatmul INT8 output "
        "best "
        "algo %d exec_time %f(ms)\n",
        INT8_gemm_cublasLt_INT8Output_best_algo.algoId, time_used);
    tmp_str = string(tmp_cstr);
    gemm_test_result += tmp_str;
  }

  if (Cdesc2) cublasLtMatrixLayoutDestroy(Cdesc2);
  if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
  if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
  if (operationDesc2) cublasLtMatmulDescDestroy(operationDesc2);
  if (workSpace2) cudaFree(workSpace2);

  checkCudaAPIErrors(cudaEventDestroy(start));
  checkCudaAPIErrors(cudaEventDestroy(stop));
}

string launch_gemm_test(int m, int n, int k) {
  gemm_test_result = "";
  memset(tmp_cstr, 0, sizeof tmp_cstr);
  sprintf(tmp_cstr, "m %d ; n %d ; k %d\n", m, n, k);
  tmp_str = string(tmp_cstr);
  gemm_test_result += tmp_str;

  half* h_half_A = NULL;
  half* h_half_A_transpose = NULL;
  half* h_half_B_transpose = NULL;
  half* h_half_B = NULL;
  int8_t* h_int8_A = NULL;
  int8_t* h_int8_B = NULL;
  int8_t* h_int8_A_transpose = NULL;
  int8_t* h_int8_B_transpose = NULL;
  float* h_float_C = NULL;

  half* d_half_A = NULL;
  half* d_half_A_transpose = NULL;
  half* d_half_B_transpose = NULL;
  half* d_half_B = NULL;
  half* d_half_C = NULL;
  int8_t* d_int8_A = NULL;
  int8_t* d_int8_B = NULL;
  int8_t* d_int8_A_transpose = NULL;
  int8_t* d_int8_B_transpose = NULL;
  int8_t* d_int8_C = NULL;
  int8_t* d_int8_C_col_major = NULL;
  int32_t* d_int_C = NULL;
  int32_t* d_int_C2 = NULL;
  float* d_float_C = NULL;

  int devID = 0;

  cudaSetDevice(devID);
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, devID);
  memset(tmp_cstr, 0, sizeof tmp_cstr);
  sprintf(tmp_cstr, "Device : %s, compute SM %d.\n", devProp.name,
          devProp.major * 10 + devProp.minor);
  tmp_str = string(tmp_cstr);
  gemm_test_result += tmp_str;

  if (devProp.major == 7 && devProp.minor == 5) {
    RUN_WITH_A100 = 0;
  } else if (devProp.major >= 8) {
    RUN_WITH_A100 = 1;
  } else {
    throw std::runtime_error("This device does not support INT8 gemm.");
  }

  cudaStream_t stream = 0;

  // allocate memory
  h_half_A = (half*)malloc(sizeof(half) * m * k);
  h_half_A_transpose = (half*)malloc(sizeof(half) * m * k);
  h_half_B_transpose = (half*)malloc(sizeof(half) * n * k);
  h_half_B = (half*)malloc(sizeof(half) * n * k);
  h_float_C = (float*)malloc(sizeof(float) * n * m);
  h_int8_A = (int8_t*)malloc(sizeof(int8_t) * m * k);
  h_int8_B = (int8_t*)malloc(sizeof(int8_t) * n * k);
  h_int8_A_transpose = (int8_t*)malloc(sizeof(int8_t) * m * k);
  h_int8_B_transpose = (int8_t*)malloc(sizeof(int8_t) * n * k);

  checkCudaAPIErrors(cudaMalloc((void**)&d_half_A, sizeof(half) * m * k));
  checkCudaAPIErrors(cudaMalloc((void**)&d_half_B, sizeof(half) * n * k));
  checkCudaAPIErrors(
      cudaMalloc((void**)&d_half_A_transpose, sizeof(half) * m * k));
  checkCudaAPIErrors(
      cudaMalloc((void**)&d_half_B_transpose, sizeof(half) * n * k));
  checkCudaAPIErrors(cudaMalloc((void**)&d_half_C, sizeof(half) * m * n));
  checkCudaAPIErrors(cudaMalloc((void**)&d_int8_A, sizeof(int8_t) * m * k));
  checkCudaAPIErrors(cudaMalloc((void**)&d_int8_B, sizeof(int8_t) * n * k));
  checkCudaAPIErrors(
      cudaMalloc((void**)&d_int8_A_transpose, sizeof(int8_t) * m * k));
  checkCudaAPIErrors(
      cudaMalloc((void**)&d_int8_B_transpose, sizeof(int8_t) * n * k));
  checkCudaAPIErrors(cudaMalloc((void**)&d_int_C, sizeof(int) * m * n));
  checkCudaAPIErrors(cudaMalloc((void**)&d_int_C2, sizeof(int) * m * n));
  checkCudaAPIErrors(cudaMalloc((void**)&d_float_C, sizeof(float) * m * n));
  checkCudaAPIErrors(cudaMalloc((void**)&d_int8_C, sizeof(int8_t) * m * n));
  checkCudaAPIErrors(
      cudaMalloc((void**)&d_int8_C_col_major, sizeof(int8_t) * m * n));

  cublasHandle_t handle;
  checkcuBlasError(cublasCreate(&handle));
  cublasLtHandle_t ltHandle;
  cublasLtCreate(&ltHandle);

  float time_used = 0.0;  // ms

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cublasStatus_t cublasStat;

  // step 1: initialize A and B, do gemm in CPU
  matInit(h_half_A, h_int8_A, m * k);  // m*k
  matInit(h_half_B, h_int8_B, n * k);  // n*k
  transpose(h_half_B_transpose, h_half_B, n, k);
  transpose(h_half_A_transpose, h_half_A, m, k);
  // matMul(n, m, k, h_half_B, n, h_half_A_transpose, k, h_float_C, n);

  transpose(h_int8_A_transpose, h_int8_A, m, k);
  transpose(h_int8_B_transpose, h_int8_B, n, k);

  cudaMemcpy(d_half_A, h_half_A, m * k * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_half_B, h_half_B, n * k * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_int8_A, h_int8_A, m * k * sizeof(int8_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_int8_B, h_int8_B, n * k * sizeof(int8_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_int8_A_transpose, h_int8_A_transpose, m * k * sizeof(int8_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_int8_B_transpose, h_int8_B_transpose, n * k * sizeof(int8_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_half_A_transpose, h_half_A_transpose, m * k * sizeof(half),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_half_B_transpose, h_half_B_transpose, n * k * sizeof(half),
             cudaMemcpyHostToDevice);

  cublasLtMatmulAlgo_info INT8_gemm_cublasLt_INT8Output_best_algo,
      INT8_gemm_cublasLt_INT8Output_COLMajor_best_algo;

  int alphaI = 1;
  int betaI = 0;
  half alpha = (half)1.0f;
  half beta = (half)0.f;

  // step 2.1 : find the best algorithms for int8 NT gemm with COL32_2R_4R4
  float alpha2 = 1.0f;
  float beta2 = 0.f;
  LtIgemmCustomFindINT8OutputCOL32_2R_4R4(
      ltHandle, n, m, k, &alpha2, d_int8_B, d_int8_A, &beta2, d_int8_C,
      INT8_gemm_cublasLt_INT8Output_best_algo);

  // step 2.2 : find the best algorithms for int8 TN gemm with COLMajor
  LtIgemmCustomFindINT8OutputColMajor(
      ltHandle, n, m, k, &alpha2, d_int8_B_transpose, d_int8_A_transpose,
      &beta2, d_int8_C_col_major,
      INT8_gemm_cublasLt_INT8Output_COLMajor_best_algo);

  // step 3 : do gemm
  // step 3.1 : FP16 NN-gemm + cublas (using tensor core)
  int iters = 10;
  cublasStat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                            d_half_B, CUDA_R_16F, n, d_half_A_transpose,
                            CUDA_R_16F, k, &beta, d_half_C, CUDA_R_16F, n,
                            CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cudaEventRecord(start, 0);
  for (int t = 1; t < iters; t++) {
    cublasStat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                              d_half_B, CUDA_R_16F, n, d_half_A_transpose,
                              CUDA_R_16F, k, &beta, d_half_C, CUDA_R_16F, n,
                              CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(start);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_used, start, stop);
  time_used /= (iters - 1);
  if (cublasStat == CUBLAS_STATUS_SUCCESS) {
    memset(tmp_cstr, 0, sizeof tmp_cstr);
    sprintf(tmp_cstr, "FP16 NN-gemm cublasGemmEx exec_time %f(ms)\n",
            time_used);
    tmp_str = string(tmp_cstr);
    gemm_test_result += tmp_str;
  }

  // step 3.2 : INT8 + gemm + COL32_2R_4R4 + cublasLt + INT8 output (using
  // tensor core)
  alpha2 = 127.0f / getAMax(d_half_C, m * n);
  int8gemm_with_cublasLtMatmul_ORDER_COL32_2R_4R4(
      n, m, k, alpha2, beta2, d_int8_B, d_int8_A, d_int8_C,
      INT8_gemm_cublasLt_INT8Output_best_algo, ltHandle, stream);

  // step 3.3 : INT8 + gemm + COL-MAJOR + cublasLt + INT8 output (using tensor
  // core)
  alpha2 = 127.0f / getAMax(d_half_C, m * n);
  int8gemm_with_cublasLtMatmul_COL_MAJOR(
      n, m, k, alpha2, beta2, d_int8_B_transpose, d_int8_A_transpose,
      d_int8_C_col_major, INT8_gemm_cublasLt_INT8Output_COLMajor_best_algo,
      ltHandle, stream);

  // step 4: check result
  // checkMat2(h_float_C, d_half_C, m * n, "d_half_C");
  // checkMat3(h_float_C, d_int8_C, m * n, alpha2, "d_int8_C");
  // checkMat3(h_float_C, d_int8_C_col_major, m * n, alpha2,
  // "d_int8_C_col_major");

  memset(tmp_cstr, 0, sizeof tmp_cstr);
  sprintf(tmp_cstr,
          ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
  tmp_str = string(tmp_cstr);
  gemm_test_result += tmp_str;

  // step 5: free memory
  free(h_half_A);
  free(h_half_A_transpose);
  free(h_half_B_transpose);
  free(h_half_B);
  free(h_int8_A);
  free(h_int8_B);
  free(h_int8_A_transpose);
  free(h_int8_B_transpose);
  free(h_float_C);

  checkCudaAPIErrors(cudaFree(d_half_A));
  checkCudaAPIErrors(cudaFree(d_half_A_transpose));
  checkCudaAPIErrors(cudaFree(d_half_B_transpose));
  checkCudaAPIErrors(cudaFree(d_half_B));
  checkCudaAPIErrors(cudaFree(d_half_C));
  checkCudaAPIErrors(cudaFree(d_int8_A));
  checkCudaAPIErrors(cudaFree(d_int8_B));
  checkCudaAPIErrors(cudaFree(d_int8_A_transpose));
  checkCudaAPIErrors(cudaFree(d_int8_B_transpose));
  checkCudaAPIErrors(cudaFree(d_int_C));
  checkCudaAPIErrors(cudaFree(d_int_C2));
  checkCudaAPIErrors(cudaFree(d_float_C));
  checkCudaAPIErrors(cudaFree(d_int8_C));
  checkCudaAPIErrors(cudaFree(d_int8_C_col_major));

  checkCudaAPIErrors(cudaEventDestroy(start));
  checkCudaAPIErrors(cudaEventDestroy(stop));
  checkcuBlasError(cublasDestroy(handle));
  checkcuBlasError(cublasLtDestroy(ltHandle));

  return gemm_test_result;
}
