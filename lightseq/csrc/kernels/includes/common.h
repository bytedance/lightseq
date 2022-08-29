#pragma once
#include <stdexcept>

#include "cuda_util.h"
#include <cuda.h>
#include <cuda_fp16.h>

namespace lightseq {

const unsigned int WARP_REDUCE_MASK = 0xffffffff;
const unsigned int WARP_SIZE = 32;
const float CUDA_FLOAT_INF_NEG = -100000000.f;  // FIXME later
const float CUDA_FLOAT_INF_POS = 100000000.f;   // FIXME later
const int CUDA_INT_INF = 2147483647;
const int MAX_THREADS = 1024;

template <typename T>
__forceinline__ __device__ T gelu(T x) {
  float cdf =
      0.5f *
      (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
  return x * cdf;
}

/* fp16 gelu */
template <>
__forceinline__ __device__ half2 gelu<half2>(half2 val) {
  half2 val_pow3 = __hmul2(val, __hmul2(val, val));
  float2 tmp_pow = __half22float2(val_pow3);
  float2 tmp = __half22float2(val);

  tmp.x =
      0.5f *
      (1.0f + tanhf((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
  tmp.y =
      0.5f *
      (1.0f + tanhf((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
  return __hmul2(val, __float22half2_rn(tmp));
}

template <typename T>
__forceinline__ __device__ T warpReduceSum(T val) {
  for (int mask = (WARP_SIZE >> 1); mask > 0; mask >>= 1)
    val += __shfl_xor_sync(WARP_REDUCE_MASK, val, mask, WARP_SIZE);
  return val;
}

template <typename T>
__forceinline__ __device__ T warpReduceMax(T val) {
  for (int mask = (WARP_SIZE >> 1); mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(WARP_REDUCE_MASK, val, mask, WARP_SIZE));
  return val;
}

template <typename T>
__forceinline__ __device__ T warpReduceMin(T val) {
  for (int mask = (WARP_SIZE >> 1); mask > 0; mask >>= 1)
    val = min(val, __shfl_xor_sync(WARP_REDUCE_MASK, val, mask, WARP_SIZE));
  return val;
}

/* Calculate the sum of all elements in a block */
template <typename T>
__forceinline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if (lane == 0) shared[wid] = val;
  __syncthreads();

  // val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)0.0f;
  val = (threadIdx.x < ((blockDim.x + 31) >> 5)) ? shared[lane] : (T)0.0f;
  val = warpReduceSum<T>(val);
  return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
__forceinline__ __device__ T blockReduceMax(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceMax<T>(val);

  if (lane == 0) shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < ((blockDim.x + 31) >> 5)) ? shared[lane]
                                                 : CUDA_FLOAT_INF_NEG;
  val = warpReduceMax<T>(val);
  return val;
}

/* Calculate the minimum of all elements in a block */
template <typename T>
__forceinline__ __device__ T blockReduceMin(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceMin<T>(val);

  if (lane == 0) shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < ((blockDim.x + 31) >> 5)) ? shared[lane]
                                                 : CUDA_FLOAT_INF_POS;
  val = warpReduceMin<T>(val);
  return val;
}

/* Calculate the rough topk-th value in a block, rough but safe */
template <typename T, int K>
__forceinline__ __device__ T blockRoughTopK(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;
  val = warpReduceMax(val);

  if (lane == 0) shared[wid] = val;
  __syncthreads();

  // we do not care about result of threadIdx.x bigger than (blockDim.x >> 5)
  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0;

  // K should be 2, 4, 6, 8, 16 or 32
  for (int mask = 16; mask >= K; mask >>= 1)
    val = max(val, __shfl_xor_sync(WARP_REDUCE_MASK, val, mask, 32));
  for (int mask = (K >> 1); mask > 0; mask >>= 1)
    val = min(val, __shfl_xor_sync(WARP_REDUCE_MASK, val, mask, 32));

  return val;
}

/* Convert 3-dim tensor index into vector index */
__forceinline__ __host__ __device__ int targetid_3dim(int id1, int id2, int id3,
                                                      int dim2, int dim3) {
  return id1 * dim2 * dim3 + id2 * dim3 + id3;
}

/* Convert 4-dim tensor index into vector index */
__forceinline__ __host__ __device__ int targetid_4dim(int id1, int id2, int id3,
                                                      int id4, int dim2,
                                                      int dim3, int dim4) {
  // return id1*(dim2*dim3*dim4) + id2*(dim3*dim4) + id3*dim4 + id4;
  int res = id4;

  int ld = dim4;
  res += id3 * ld;

  ld *= dim3;
  res += id2 * ld;

  ld *= dim2;
  res += id1 * ld;

  return res;
}

/* Convert 5-dim tensor index into vector index */
__forceinline__ __host__ __device__ int targetid_5dim(int id1, int id2, int id3,
                                                      int id4, int id5,
                                                      int dim2, int dim3,
                                                      int dim4, int dim5) {
  // return id1*(dim2*dim3*dim4*dim5) + id2*(dim3*dim4*dim5) + id3*(dim4*dim5) +
  // id4*dim5 + dim5;
  int res = id5;

  int ld = dim5;
  res += id4 * ld;

  ld *= dim4;
  res += id3 * ld;

  ld *= dim3;
  res += id2 * ld;

  ld *= dim2;
  res += id1 * ld;

  return res;
}

/* Convert 6-dim tensor index into vector index */
__forceinline__ __host__ __device__ int targetid_6dim(int id1, int id2, int id3,
                                                      int id4, int id5, int id6,
                                                      int dim2, int dim3,
                                                      int dim4, int dim5,
                                                      int dim6) {
  // return id1*(dim2*dim3*dim4*dim5*dim6) + id2*(dim3*dim4*dim5*dim6) +
  // id3*(dim4*dim5*dim6) + id4*(dim5*dim6) + id5*dim6 + id6;
  int res = id6;

  int ld = dim6;
  res += id5 * ld;

  ld *= dim5;
  res += id4 * ld;

  ld *= dim4;
  res += id3 * ld;

  ld *= dim3;
  res += id2 * ld;

  ld *= dim2;
  res += id1 * ld;

  return res;
}

/* Convert half2 into float2, mask inf and -inf */
__forceinline__ __host__ __device__ float safe_half_to_float(half hval) {
  return fmax(fmin(100000.f, __half2float(hval)), -100000.f);
}

/* Convert half2 into float2, mask inf and -inf */
__forceinline__ __host__ __device__ float2 safe_half2_to_float2(half2 vhalf2) {
  float2 vfloat2 = __half22float2(vhalf2);
  vfloat2.x = fmax(fmin(100000.f, vfloat2.x), -100000.f);
  vfloat2.y = fmax(fmin(100000.f, vfloat2.y), -100000.f);
  return vfloat2;
}

/* flat_ndim and decompose_ndim. index transform copy from training */
/* Convert 2-dim tensor index into vector index */
__forceinline__ __host__ __device__ int flat_2dim(int id1, int id2, int dim2) {
  return id1 * dim2 + id2;
}

/* Convert 3-dim tensor index into vector index */
__forceinline__ __host__ __device__ int flat_3dim(int id1, int id2, int id3,
                                                  int dim2, int dim3) {
  return id1 * dim2 * dim3 + id2 * dim3 + id3;
}

/* Convert 4-dim tensor index into vector index */
__forceinline__ __host__ __device__ int flat_4dim(int id1, int id2, int id3,
                                                  int id4, int dim2, int dim3,
                                                  int dim4) {
  // return id1*(dim2*dim3*dim4) + id2*(dim3*dim4) + id3*dim4 + id4;
  int res = id4;

  int ld = dim4;
  res += id3 * ld;

  ld *= dim3;
  res += id2 * ld;

  ld *= dim2;
  res += id1 * ld;

  return res;
}

/* Convert 5-dim tensor index into vector index */
__forceinline__ __host__ __device__ int flat_5dim(int id1, int id2, int id3,
                                                  int id4, int id5, int dim2,
                                                  int dim3, int dim4,
                                                  int dim5) {
  // return id1*(dim2*dim3*dim4*dim5) + id2*(dim3*dim4*dim5) + id3*(dim4*dim5) +
  // id4*dim5 + dim5;
  int res = id5;

  int ld = dim5;
  res += id4 * ld;

  ld *= dim4;
  res += id3 * ld;

  ld *= dim3;
  res += id2 * ld;

  ld *= dim2;
  res += id1 * ld;

  return res;
}

/* Convert 6-dim tensor index into vector index */
__forceinline__ __host__ __device__ int flat_6dim(int id1, int id2, int id3,
                                                  int id4, int id5, int id6,
                                                  int dim2, int dim3, int dim4,
                                                  int dim5, int dim6) {
  // return id1*(dim2*dim3*dim4*dim5*dim6) + id2*(dim3*dim4*dim5*dim6) +
  // id3*(dim4*dim5*dim6) + id4*(dim5*dim6) + id5*dim6 + id6;
  int res = id6;

  int ld = dim6;
  res += id5 * ld;

  ld *= dim5;
  res += id4 * ld;

  ld *= dim4;
  res += id3 * ld;

  ld *= dim3;
  res += id2 * ld;

  ld *= dim2;
  res += id1 * ld;

  return res;
}

/* row major index to col32 index */
__forceinline__ __host__ __device__ int row_major2flat_col32(int row_id,
                                                             int col_id,
                                                             int row_size,
                                                             int col_size) {
  return ((col_id & 0xffffe0) * row_size) + (row_id << 5) + (col_id & 31);
}

/* Convert vector index to 6-dim tensor index */
__forceinline__ __host__ __device__ void decompose_6dim(
    int src, int dim1, int dim2, int dim3, int dim4, int dim5, int *id0,
    int *id1, int *id2, int *id3, int *id4, int *id5) {
  *id5 = src % dim5;
  src /= dim5;

  *id4 = src % dim4;
  src /= dim4;

  *id3 = src % dim3;
  src /= dim3;

  *id2 = src % dim2;
  src /= dim2;

  *id1 = src % dim1;
  *id0 = src / dim1;
}

/* Convert vector index to 5-dim tensor index */
__forceinline__ __host__ __device__ void decompose_5dim(int src, int dim1,
                                                        int dim2, int dim3,
                                                        int dim4, int *id0,
                                                        int *id1, int *id2,
                                                        int *id3, int *id4) {
  *id4 = src % dim4;
  src /= dim4;

  *id3 = src % dim3;
  src /= dim3;

  *id2 = src % dim2;
  src /= dim2;

  *id1 = src % dim1;
  *id0 = src / dim1;
}

/* Convert vector index to 4-dim tensor index */
__forceinline__ __host__ __device__ void decompose_4dim(int src, int dim1,
                                                        int dim2, int dim3,
                                                        int *id0, int *id1,
                                                        int *id2, int *id3) {
  *id3 = src % dim3;
  src /= dim3;

  *id2 = src % dim2;
  src /= dim2;

  *id1 = src % dim1;
  *id0 = src / dim1;
}

/* Convert vector index to 3-dim tensor index */
__forceinline__ __host__ __device__ void decompose_3dim(int src, int dim1,
                                                        int dim2, int *id0,
                                                        int *id1, int *id2) {
  *id2 = src % dim2;
  src /= dim2;

  *id1 = src % dim1;
  *id0 = src / dim1;
}

/* Convert vector index to 2-dim tensor index */
__forceinline__ __host__ __device__ void decompose_2dim(int src, int dim1,
                                                        int *id0, int *id1) {
  *id1 = src % dim1;
  *id0 = src / dim1;
}

}  // namespace lightseq
