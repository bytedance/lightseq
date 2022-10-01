#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include "transformerKernels_int8.h"
#include "cublas_algo_map.h"

namespace lightseq {
namespace cuda {

int getSMVersion();

static bool use_ORDER_COL32_2R_4R4 = (getSMVersion() >= 80 ? true : false);

enum Layout {
  kRowMajor,
  kColMajor,
  kColMajor32,
};

void cublasLtMM_withAlgo(int* res, int batchCount, int m, int n, int k,
                         int64_t stridea, int64_t strideb, int64_t stridec,
                         const int8_t* ATransform, const int8_t* kernel,
                         cublasLtHandle_t cublasLt_handle, cudaStream_t stream,
                         bool use_ORDER_COL32_2R_4R4);

void cublasLtMM_withAlgo_i8IO(int8_t* res, int batchCount, int m, int n, int k,
                              int64_t stridea, int64_t strideb, int64_t stridec,
                              const float alpha, const int8_t* ATransform,
                              const int8_t* kernel,
                              cublasLtHandle_t cublasLt_handle,
                              cudaStream_t stream, bool use_ORDER_COL32_2R_4R4);

template <typename OutType, typename ScaleType>
void cublaslt_gemm(const int8_t* input_a, const int8_t* input_b,
                   OutType* output_c, int batch_count, int m, int n, int k,
                   int64_t stridea, int64_t strideb, int64_t stridec,
                   const ScaleType alpha, cublasLtHandle_t cublasLt_handle,
                   cudaStream_t stream);

template <typename OutType, typename ScaleType>
void cublaslt_gemm(const int8_t* input_a, const int8_t* input_b,
                   OutType* output_c, int batch_count, int m, int n, int k,
                   int64_t stridea, int64_t strideb, int64_t stridec,
                   const ScaleType alpha, cublasLtHandle_t cublasLt_handle,
                   cudaStream_t stream, cublasAlgoMap& algo_map);

inline int round_up(int v, int d) { return (v + d - 1) / d * d; }

void transform_weight_layout(const int8_t* input, int8_t* output, int row,
                             int col, Layout layout, cublasLtHandle_t lt_handle,
                             cudaStream_t stream);

template <typename T>
void quantize_weight(const T* origin_weight, int8_t* quantized_weight, int rows,
                     int cols, float quant_scale, cudaStream_t stream,
                     cublasLtHandle_t handle, Layout layout = kColMajor32);

}  // namespace cuda
}  // namespace lightseq
