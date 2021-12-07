#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include "transformerKernels_int8.h"

namespace lightseq {
namespace cuda {

const bool full_int8 = true;

// for int8 cublasLtMM with algo
// ATransform should be m*n, CUBLASLT_ORDER_COL32
// kernel should be n*k, CUBLASLT_ORDER_COL4_4R2_8C or
// CUBLASLT_ORDER_COL32_2R_4R4 res is m*n, CUBLASLT_ORDER_COL32
void cublasLtMM_withAlgo(
    int* res, int batchCount, int m, int n, int k, int64_t stridea,
    int64_t strideb, int64_t stridec, const int8_t* ATransform,
    const int8_t* kernel, cublasLtHandle_t cublasLt_handle, cudaStream_t stream,
    // std::map<std::string, cublasLtMatmulAlgo_info>& cublasLtAlgoMap,
    bool use_ORDER_COL32_2R_4R4);

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
    bool use_ORDER_COL32_2R_4R4);

inline int roundoff(int v, int d) { return (v + d - 1) / d * d; }

void transform_weight_row_major2col32t(const int8_t* input, int8_t* output,
                                       int row, int col,
                                       cublasLtHandle_t lt_handle,
                                       cudaStream_t stream);

template <typename T>
void quantize_weight_col32t(const T* origin_weight, int8_t* quantized_weight,
                            int rows, int cols, int quant_range, float clip_max,
                            cudaStream_t stream, cublasLtHandle_t handle);

}  // namespace cuda
}  // namespace lightseq
