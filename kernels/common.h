#pragma once

#include <cuda.h>
#include <cuda_fp16.h>

namespace lab {
namespace nmt {

const unsigned int WARP_REDUCE_MASK = 0xffffffff;
const unsigned int WARP_SIZE = 32;
const float CUDA_FLOAT_INF_NEG = -100000000.f; // FIXME later

template <typename T>
__forceinline__ __device__ 
T warpReduceSum(T val) {
  for (int mask = (WARP_SIZE >> 1); mask > 0; mask >>= 1)
    val += __shfl_xor_sync(WARP_REDUCE_MASK, val, mask, WARP_SIZE);
  return val;
}

template <typename T>
__forceinline__ __device__ 
T warpReduceMax(T val) {
  for (int mask = (WARP_SIZE >> 1); mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(WARP_REDUCE_MASK, val, mask, WARP_SIZE));
  return val;
}

/* Calculate the sum of all elements in a block */
template <typename T>
__forceinline__ __device__ 
T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if (lane == 0) 
    shared[wid] = val;
  __syncthreads();

  // val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)0.0f;
  val = (threadIdx.x < ((blockDim.x + 31) >> 5)) ? shared[lane] : (T)0.0f;
  val = warpReduceSum<T>(val);
  return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
__forceinline__ __device__ 
T blockReduceMax(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceMax<T>(val);

  if (lane == 0) 
    shared[wid] = val;
  __syncthreads();

  // FIXME later. provide inf value for int32, fp16
  // val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : CUDA_FLOAT_INF_NEG;
  val = (threadIdx.x < ((blockDim.x + 31) >> 5)) ? shared[lane] : CUDA_FLOAT_INF_NEG;
  val = warpReduceMax<T>(val);
  return val;
}

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

  // K should be 2, 4, 6, 8, 16, 32
  for (int mask = 16; mask >= K; mask >>= 1)
    val = max(val, __shfl_xor_sync(WARP_REDUCE_MASK, val, mask, 32));
  for (int mask = (K >> 1); mask > 0; mask >>= 1)
    val = min(val, __shfl_xor_sync(WARP_REDUCE_MASK, val, mask, 32));

  return val;
}

__forceinline__ __host__ __device__ int targetid_3dim(int id1, int id2, int id3,
                                                      int dim2, int dim3) {
  return id1 * dim2 * dim3 + id2 * dim3 + id3;
  // int res = id3;
  //
  // int ld = dim3;
  // res += id2 * ld;
  //
  // ld *= dim2;
  // res += id1 * ld;
  //
  // return res;
}

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

__forceinline__ __host__ __device__ 
float2 safe_half2_to_float2(half2 vhalf2) {
  float2 vfloat2 = __half22float2(vhalf2);
  // mask inf and -inf
  vfloat2.x = fmax(fmin(100000.f, vfloat2.x), -100000.f);
  vfloat2.y = fmax(fmin(100000.f, vfloat2.y), -100000.f);
  return vfloat2;
}

}  // namespace nmt
}  // namespace lab
