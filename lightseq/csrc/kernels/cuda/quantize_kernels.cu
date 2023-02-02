#include "kernels.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;
namespace lightseq {
namespace cuda {
__device__ __host__ int row_major2cublaslt_weight(int col_id, int row_id,
                                                  int m_32,
                                                  bool use_col32_2r_4r4) {
  if (use_col32_2r_4r4) {
    int new_col = col_id >> 5;
    int row_in_tile = row_id & 31;
    int col_in_tile = col_id & 31;
    int new_row =  // CUBLASLT_ORDER_COL32_2R_4R4
        (((row_id >> 5) << 10) +
         //(((row%8)/2*4+row/8)*2+row%2)*32+col
         (((((((row_in_tile & 7) >> 1) << 2) + (row_in_tile >> 3)) << 1) +
           (row_in_tile & 1))
          << 5) +
         col_in_tile);
    return new_col * m_32 + new_row;
  } else {
    int new_col = col_id >> 5;
    int new_row =  // CUBLASLT_ORDER_COL4_4R2_8C
                   ////row_id/8 is the number of tile of (8 rows 32 columns) --
                   /// column-major /row_id%2 is even row, otherwise odd row
                   ////col_id%COL32_/8 is the number tile of (8 rows 8 columns)
        (((((row_id >> 3) << 3) + ((row_id & 1) << 2) + ((col_id & 31) >> 3))
          << 5) +
         ////col_id%8 >= 4 is the right half of (8 rows 8 columns) tile
         ////(row_id%8/2) is (the row id of alternating 4 rows) - 1
         (((((col_id & 7) >= 4) ? 4 : 0) + ((row_id & 7) >> 1)) << 2) +
         ////col_id%4 is the id of 4 cols
         (col_id & 3));
    return new_col * m_32 + new_row;
  }
}

template <typename T>
__global__ void quantize_kernel(int8_t *q_ptr, uint8_t *clip_mask_ptr,
                                float *alpha_ptr, const T *f_ptr,
                                const T *clip_max_ptr, int numel,
                                int mask_start_bit);
template <>
__global__ void quantize_kernel<float>(int8_t *q_ptr, uint8_t *clip_mask_ptr,
                                       float *alpha_ptr, const float *f_ptr,
                                       const float *clip_max_ptr, int numel,
                                       int mask_start_bit) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 4 >= numel) return;

  float clip_max_val = clip_max_ptr[0];

  int32_t *q_weight4_ptr = reinterpret_cast<int32_t *>(q_ptr);
  const float4 *weight4_ptr = reinterpret_cast<const float4 *>(f_ptr);
  uint32_t *clip_mask4_ptr = reinterpret_cast<uint32_t *>(clip_mask_ptr);

  float4 weight4 = weight4_ptr[i];
  int8_t q_weight[4];
  uint8_t clip_mask[4];
  q_weight[0] = quantize(weight4.x, clip_max_val, clip_mask[0], mask_start_bit);
  q_weight[1] = quantize(weight4.y, clip_max_val, clip_mask[1], mask_start_bit);
  q_weight[2] = quantize(weight4.z, clip_max_val, clip_mask[2], mask_start_bit);
  q_weight[3] = quantize(weight4.w, clip_max_val, clip_mask[3], mask_start_bit);

  q_weight4_ptr[i] = reinterpret_cast<int32_t *>(q_weight)[0];
  if (clip_mask_ptr) {
    clip_mask4_ptr[i] |= reinterpret_cast<uint32_t *>(clip_mask)[0];
  }

  if (blockIdx.x == 0 && threadIdx.x == 0 && alpha_ptr) {
    float input_cmax = clip_max_val;
    float weight_cmax = clip_max_ptr[1];
    float output_cmax = clip_max_ptr[2];
    alpha_ptr[0] = input_cmax * weight_cmax / (output_cmax * kQuantRangeI8);
  }
}

template <>
__global__ void quantize_kernel<__half>(int8_t *q_ptr, uint8_t *clip_mask_ptr,
                                        float *alpha_ptr, const __half *f_ptr,
                                        const __half *clip_max_ptr, int numel,
                                        int mask_start_bit) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 8 >= numel) return;

  float clip_max_val = __half2float(clip_max_ptr[0]);

  int64_t *q_weight8_ptr = reinterpret_cast<int64_t *>(q_ptr);
  const float4 *weight8_ptr = reinterpret_cast<const float4 *>(f_ptr);
  uint64_t *clip_mask8_ptr = reinterpret_cast<uint64_t *>(clip_mask_ptr);

  float4 weight8 = weight8_ptr[i];
  int8_t q_weight[8];
  uint8_t clip_mask[8];
  __half2 *weight2_ptr = reinterpret_cast<__half2 *>(&weight8);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    q_weight[i * 2] = quantize(__half2float(weight2_ptr[i].x), clip_max_val,
                               clip_mask[i * 2], mask_start_bit);
    q_weight[i * 2 + 1] = quantize(__half2float(weight2_ptr[i].y), clip_max_val,
                                   clip_mask[i * 2 + 1], mask_start_bit);
  }

  q_weight8_ptr[i] = reinterpret_cast<int64_t *>(q_weight)[0];
  if (clip_mask_ptr) {
    clip_mask8_ptr[i] |= reinterpret_cast<uint64_t *>(clip_mask)[0];
  }
  if (blockIdx.x == 0 && threadIdx.x == 0 && alpha_ptr) {
    float input_cmax = clip_max_val;
    float weight_cmax = __half2float(clip_max_ptr[1]);
    float output_cmax = __half2float(clip_max_ptr[2]);
    alpha_ptr[0] = input_cmax * weight_cmax / (output_cmax * kQuantRangeI8);
  }
}

template <>
void launch_quantize<float>(int8_t *q_ptr, uint8_t *clip_mask_ptr,
                            float *alpha_ptr, const float *f_ptr,
                            const float *clip_max_ptr, int numel,
                            int mask_start_bit, cudaStream_t stream) {
  if (numel % 4 != 0) {
    throw std::runtime_error("violate numel % 4 = 0");
  }
  int ele_per_block = MAX_THREADS * 4;
  int grid_dim = numel / ele_per_block + 1;
  quantize_kernel<<<grid_dim + 1, MAX_THREADS, 0, stream>>>(
      q_ptr, clip_mask_ptr, alpha_ptr, f_ptr, clip_max_ptr, numel,
      mask_start_bit);
}

template <>
void launch_quantize<__half>(int8_t *q_ptr, uint8_t *clip_mask_ptr,
                             float *alpha_ptr, const __half *f_ptr,
                             const __half *clip_max_ptr, int numel,
                             int mask_start_bit, cudaStream_t stream) {
  if (numel % 8 != 0) {
    throw std::runtime_error("violate numel % 8 = 0");
  }
  int ele_per_block = MAX_THREADS * 8;
  int grid_dim = numel / ele_per_block + 1;
  quantize_kernel<<<grid_dim + 1, MAX_THREADS, 0, stream>>>(
      q_ptr, clip_mask_ptr, alpha_ptr, f_ptr, clip_max_ptr, numel,
      mask_start_bit);
}

template <typename T>
__global__ void fake_quantize_kernel(uint8_t *clip_mask_ptr, float *alpha_ptr,
                                     T *output, const T *input,
                                     const T *clip_max_ptr, int numel,
                                     int mask_start_bit, bool symmetry);
template <>
__global__ void fake_quantize_kernel<float>(
    uint8_t *clip_mask_ptr, float *alpha_ptr, float *output, const float *input,
    const float *clip_max_ptr, int numel, int mask_start_bit, bool symmetry) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 4 >= numel) return;

  float clip_max_val = clip_max_ptr[0];

  const float4 *input4_ptr = reinterpret_cast<const float4 *>(input);
  float4 *output4_ptr = reinterpret_cast<float4 *>(output);
  uint32_t *clip_mask4_ptr;
  if (clip_mask_ptr) {
    clip_mask4_ptr = reinterpret_cast<uint32_t *>(clip_mask_ptr);
  }

  float4 input4 = input4_ptr[i];

  uint8_t clip_mask[4];

  input4.x = fake_quantize(input4.x, clip_max_val, clip_mask[0], mask_start_bit,
                           symmetry);
  input4.y = fake_quantize(input4.y, clip_max_val, clip_mask[1], mask_start_bit,
                           symmetry);
  input4.z = fake_quantize(input4.z, clip_max_val, clip_mask[2], mask_start_bit,
                           symmetry);
  input4.w = fake_quantize(input4.w, clip_max_val, clip_mask[3], mask_start_bit,
                           symmetry);
  output4_ptr[i] = input4;

  if (clip_mask_ptr) {
    clip_mask4_ptr[i] |= reinterpret_cast<uint32_t *>(clip_mask)[0];
  }

  if (blockIdx.x == 0 && threadIdx.x == 0 && alpha_ptr) {
    float input_cmax = clip_max_ptr[0];
    float weight_cmax = clip_max_val;
    float output_cmax = clip_max_ptr[2];
    alpha_ptr[0] = input_cmax * weight_cmax / (output_cmax * kQuantRangeI8);
  }
}

template <>
__global__ void fake_quantize_kernel<__half>(uint8_t *clip_mask_ptr,
                                             float *alpha_ptr, __half *output,
                                             const __half *input,
                                             const __half *clip_max_ptr,
                                             int numel, int mask_start_bit,
                                             bool symmetry) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 8 >= numel) return;

  float clip_max_val = __half2float(clip_max_ptr[0]);

  const float4 *input8_ptr = reinterpret_cast<const float4 *>(input);
  float4 *output8_ptr = reinterpret_cast<float4 *>(output);
  uint64_t *clip_mask8_ptr;
  if (clip_mask_ptr) {
    clip_mask8_ptr = reinterpret_cast<uint64_t *>(clip_mask_ptr);
  }
  float4 input8 = input8_ptr[i];

  uint8_t clip_mask[8];
  __half2 *input2_ptr = reinterpret_cast<__half2 *>(&input8);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    input2_ptr[i].x = fake_quantize(__half2float(input2_ptr[i].x), clip_max_val,
                                    clip_mask[i * 2], mask_start_bit, symmetry);
    input2_ptr[i].y =
        fake_quantize(__half2float(input2_ptr[i].y), clip_max_val,
                      clip_mask[i * 2 + 1], mask_start_bit, symmetry);
  }

  output8_ptr[i] = input8;
  if (clip_mask_ptr) {
    clip_mask8_ptr[i] |= reinterpret_cast<uint64_t *>(clip_mask)[0];
  }
  if (blockIdx.x == 0 && threadIdx.x == 0 && alpha_ptr) {
    float input_cmax = __half2float(clip_max_ptr[0]);
    float weight_cmax = clip_max_val;
    float output_cmax = __half2float(clip_max_ptr[2]);
    alpha_ptr[0] = input_cmax * weight_cmax / (output_cmax * kQuantRangeI8);
  }
}

template <>
void launch_fake_quantize<float>(uint8_t *clip_mask_ptr, float *alpha_ptr,
                                 float *output, const float *input,
                                 const float *clip_max_ptr, int numel,
                                 int mask_start_bit, cudaStream_t stream,
                                 bool symmetry) {
  if (numel % 4 != 0) {
    throw std::runtime_error("violate numel % 4 = 0");
  }
  int ele_per_block = MAX_THREADS * 4;
  int grid_dim = numel / ele_per_block + 1;
  fake_quantize_kernel<<<grid_dim + 1, MAX_THREADS, 0, stream>>>(
      clip_mask_ptr, alpha_ptr, output, input, clip_max_ptr, numel,
      mask_start_bit, symmetry);
}

template <>
void launch_fake_quantize<__half>(uint8_t *clip_mask_ptr, float *alpha_ptr,
                                  __half *output, const __half *input,
                                  const __half *clip_max_ptr, int numel,
                                  int mask_start_bit, cudaStream_t stream,
                                  bool symmetry) {
  if (numel % 8 != 0) {
    throw std::runtime_error("violate numel % 8 = 0");
  }
  int ele_per_block = MAX_THREADS * 8;
  int grid_dim = numel / ele_per_block + 1;
  fake_quantize_kernel<<<grid_dim + 1, MAX_THREADS, 0, stream>>>(
      clip_mask_ptr, alpha_ptr, output, input, clip_max_ptr, numel,
      mask_start_bit, symmetry);
}

__global__ void dequantize_kernel(float *f_ptr, const int8_t *q_ptr,
                                  const float *clip_max_ptr, int numel,
                                  int mask_start_bit) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 4 >= numel) return;

  float clip_max_val = clip_max_ptr[0];

  const int32_t *q_weight4_ptr = reinterpret_cast<const int32_t *>(q_ptr);
  float4 *weight4_ptr = reinterpret_cast<float4 *>(f_ptr);

  float4 weight4;
  int32_t q_weight_i32 = q_weight4_ptr[i];
  int8_t *q_weight = reinterpret_cast<int8_t *>(&q_weight_i32);
  weight4.x = dequantize(q_weight[0], clip_max_val);
  weight4.y = dequantize(q_weight[1], clip_max_val);
  weight4.z = dequantize(q_weight[2], clip_max_val);
  weight4.w = dequantize(q_weight[3], clip_max_val);

  weight4_ptr[i] = weight4;
}

__global__ void dequantize_kernel(__half *f_ptr, const int8_t *q_ptr,
                                  const __half *clip_max_ptr, int numel,
                                  int mask_start_bit) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 8 >= numel) return;

  float clip_max_val = __half2float(clip_max_ptr[0]);

  const int64_t *q_weight8_ptr = reinterpret_cast<const int64_t *>(q_ptr);
  float4 *weight8_ptr = reinterpret_cast<float4 *>(f_ptr);

  float4 weight8;
  int64_t q_weight_i64 = q_weight8_ptr[i];
  int8_t *q_weight = reinterpret_cast<int8_t *>(&q_weight_i64);
  __half2 *weight2_ptr = reinterpret_cast<__half2 *>(&weight8);
#pragma unroll
  for (int j = 0; j < 4; j++) {
    weight2_ptr[j].x = __float2half(dequantize(q_weight[j * 2], clip_max_val));
    weight2_ptr[j].y =
        __float2half(dequantize(q_weight[j * 2 + 1], clip_max_val));
  }

  weight8_ptr[i] = weight8;
}

template <>
void launch_dequantize<float>(float *f_ptr, const int8_t *q_ptr,
                              const float *clip_max_ptr, int numel,
                              int mask_start_bit, cudaStream_t stream) {
  if (numel % 4 != 0) {
    throw std::runtime_error("violate numel % 4 = 0");
  }
  int ele_per_block = MAX_THREADS * 4;
  int grid_dim = numel / ele_per_block + 1;
  dequantize_kernel<<<grid_dim + 1, MAX_THREADS, 0, stream>>>(
      f_ptr, q_ptr, clip_max_ptr, numel, mask_start_bit);
}

template <>
void launch_dequantize<__half>(__half *f_ptr, const int8_t *q_ptr,
                               const __half *clip_max_ptr, int numel,
                               int mask_start_bit, cudaStream_t stream) {
  if (numel % 8 != 0) {
    throw std::runtime_error("violate numel % 8 = 0");
  }
  int ele_per_block = MAX_THREADS * 8;
  int grid_dim = numel / ele_per_block + 1;
  dequantize_kernel<<<grid_dim + 1, MAX_THREADS, 0, stream>>>(
      f_ptr, q_ptr, clip_max_ptr, numel, mask_start_bit);
}

__global__ void quantize_bwd_kernel(float *f_ptr, float *cmax_grad_ptr,
                                    const uint8_t *clip_mask_ptr, int numel,
                                    int mask_start_bit) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 4 >= numel) return;

  float4 *weight4_ptr = reinterpret_cast<float4 *>(f_ptr);
  const uint32_t *clip_mask4_ptr =
      reinterpret_cast<const uint32_t *>(clip_mask_ptr);

  float4 weight4 = weight4_ptr[i];
  uint32_t clip_mask4 = clip_mask4_ptr[i];
  uint8_t *clip_mask = reinterpret_cast<uint8_t *>(&clip_mask4);
  float accum_cmax_grad = 0;

  accum_cmax_grad += weight4.x * is_max_min_mask(clip_mask[0], mask_start_bit);
  accum_cmax_grad += weight4.y * is_max_min_mask(clip_mask[1], mask_start_bit);
  accum_cmax_grad += weight4.z * is_max_min_mask(clip_mask[2], mask_start_bit);
  accum_cmax_grad += weight4.w * is_max_min_mask(clip_mask[3], mask_start_bit);

  weight4.x = weight4.x * (is_max_min_mask(clip_mask[0], mask_start_bit) == 0);
  weight4.y = weight4.y * (is_max_min_mask(clip_mask[0], mask_start_bit) == 0);
  weight4.z = weight4.z * (is_max_min_mask(clip_mask[0], mask_start_bit) == 0);
  weight4.w = weight4.w * (is_max_min_mask(clip_mask[0], mask_start_bit) == 0);

  weight4_ptr[i] = weight4;

  __shared__ float reduction_s[MAX_THREADS / 32];
  cg::thread_block cta = cg::this_thread_block();
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);
  float reduce_cmax_grad = cg::reduce(tile, accum_cmax_grad, cg::plus<float>());
  if (tile.thread_rank() == 0) {
    reduction_s[tile.meta_group_rank()] = reduce_cmax_grad;
  }
  cg::sync(cta);
  if (cta.thread_rank() == 0) {
    reduce_cmax_grad = 0;
    for (int i = 0; i < tile.meta_group_size(); ++i) {
      reduce_cmax_grad += reduction_s[i];
    }
    atomicAdd(cmax_grad_ptr, reduce_cmax_grad);
  }
}

__global__ void quantize_bwd_kernel(__half *f_ptr, __half *cmax_grad_ptr,
                                    const uint8_t *clip_mask_ptr, int numel,
                                    int mask_start_bit) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 8 >= numel) return;

  float4 *weight8_ptr = reinterpret_cast<float4 *>(f_ptr);
  const uint64_t *clip_mask8_ptr =
      reinterpret_cast<const uint64_t *>(clip_mask_ptr);

  float4 weight8 = weight8_ptr[i];
  uint64_t clip_mask8 = clip_mask8_ptr[i];
  uint8_t *clip_mask = reinterpret_cast<uint8_t *>(&clip_mask8);
  __half2 *weight2_ptr = reinterpret_cast<__half2 *>(&weight8);
  float accum_cmax_grad;

#pragma unroll
  for (int i = 0; i < 4; i++) {
    accum_cmax_grad += __half2float(weight2_ptr[i].x) *
                       is_max_min_mask(clip_mask[i * 2], mask_start_bit);
    accum_cmax_grad += __half2float(weight2_ptr[i].y) *
                       is_max_min_mask(clip_mask[i * 2 + 1], mask_start_bit);
    weight2_ptr[i].x =
        __float2half(__half2float(weight2_ptr[i].x) *
                     (is_max_min_mask(clip_mask[i * 2], mask_start_bit) == 0));
    weight2_ptr[i].y = __float2half(
        __half2float(weight2_ptr[i].y) *
        (is_max_min_mask(clip_mask[i * 2 + 1], mask_start_bit) == 0));
  }

  weight8_ptr[i] = weight8;

  __shared__ float reduction_s[MAX_THREADS / 32];
  cg::thread_block cta = cg::this_thread_block();
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);
  float reduce_cmax_grad = cg::reduce(tile, accum_cmax_grad, cg::plus<float>());
  if (tile.thread_rank() == 0) {
    reduction_s[tile.meta_group_rank()] = reduce_cmax_grad;
  }
  cg::sync(cta);
  if (cta.thread_rank() == 0) {
    reduce_cmax_grad = 0;
    for (int i = 0; i < tile.meta_group_size(); ++i) {
      reduce_cmax_grad += reduction_s[i];
    }
    atomicAdd(cmax_grad_ptr, __float2half(reduce_cmax_grad));
  }
}

template <>
void launch_quantize_bwd<float>(float *grad_ptr, float *cmax_grad_ptr,
                                const uint8_t *clip_mask_ptr, int numel,
                                int mask_start_bit, cudaStream_t stream) {
  if (numel % 4 != 0) {
    throw std::runtime_error("violate numel % 4 = 0");
  }
  int ele_per_block = MAX_THREADS * 4;
  int grid_dim = numel / ele_per_block + 1;
  quantize_bwd_kernel<<<grid_dim + 1, MAX_THREADS, 0, stream>>>(
      grad_ptr, cmax_grad_ptr, clip_mask_ptr, numel, mask_start_bit);
}

template <>
void launch_quantize_bwd<__half>(__half *grad_ptr, __half *cmax_grad_ptr,
                                 const uint8_t *clip_mask_ptr, int numel,
                                 int mask_start_bit, cudaStream_t stream) {
  if (numel % 8 != 0) {
    throw std::runtime_error("violate numel % 8 = 0");
  }
  int ele_per_block = MAX_THREADS * 8;
  int grid_dim = numel / ele_per_block + 1;
  quantize_bwd_kernel<<<grid_dim + 1, MAX_THREADS, 0, stream>>>(
      grad_ptr, cmax_grad_ptr, clip_mask_ptr, numel, mask_start_bit);
}

template <typename T>
__global__ void d_cmax_kernel(T *grad, T *grad_cmax, const uint8_t *cmask,
                              int numel, int mask_start_bit);
template <>
__global__ void d_cmax_kernel<float>(float *grad, float *grad_cmax,
                                     const uint8_t *cmask, int numel,
                                     int mask_start_bit) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 4 >= numel) return;

  __shared__ float block_grad_cmax;

  float4 *grad4 = reinterpret_cast<float4 *>(grad);
  const uint32_t *cmask4 = reinterpret_cast<const uint32_t *>(cmask);

  float4 grad4_i = grad4[i];
  uint32_t cmask4_i = cmask4[i];
  uint8_t *cmask_i = reinterpret_cast<uint8_t *>(&cmask4_i);
  float thread_grad_cmax = 0;
  float temp_cmax_g;

  clip_bwd(grad4_i.x, temp_cmax_g, grad4_i.x, cmask_i[0], mask_start_bit);
  thread_grad_cmax += temp_cmax_g;
  clip_bwd(grad4_i.y, temp_cmax_g, grad4_i.y, cmask_i[1], mask_start_bit);
  thread_grad_cmax += temp_cmax_g;
  clip_bwd(grad4_i.z, temp_cmax_g, grad4_i.z, cmask_i[2], mask_start_bit);
  thread_grad_cmax += temp_cmax_g;
  clip_bwd(grad4_i.w, temp_cmax_g, grad4_i.w, cmask_i[3], mask_start_bit);
  thread_grad_cmax += temp_cmax_g;

  grad4[i] = grad4_i;
  if (grad_cmax) {
    if (threadIdx.x == 0) {
      block_grad_cmax = 0;
    }
    __syncthreads();
    if (thread_grad_cmax != 0) {
      atomicAdd(&block_grad_cmax, thread_grad_cmax);
    }
    __syncthreads();
    if (threadIdx.x == 0 && block_grad_cmax != 0) {
      atomicAdd(grad_cmax, block_grad_cmax);
    }
  }
}

template <>
__global__ void d_cmax_kernel<__half>(__half *grad, __half *grad_cmax,
                                      const uint8_t *cmask, int numel,
                                      int mask_start_bit) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 8 >= numel) return;

  __shared__ float block_grad_cmax;

  float4 *grad8 = reinterpret_cast<float4 *>(grad);
  const uint64_t *cmask8 = reinterpret_cast<const uint64_t *>(cmask);

  float4 grad8_i = grad8[i];
  uint64_t cmask8_i = cmask8[i];
  uint8_t *cmask_i = reinterpret_cast<uint8_t *>(&cmask8_i);
  __half2 *grad2_i = reinterpret_cast<__half2 *>(&grad8_i);

  float thread_grad_cmax = 0;
  float temp_cmax_g;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    // thread_grad_cmax += is_max_min_mask(cmask_i[i * 2], mask_start_bit);
    // thread_grad_cmax += is_max_min_mask(cmask_i[i * 2 + 1], mask_start_bit);
    clip_bwd(grad2_i[i].x, temp_cmax_g, grad2_i[i].x, cmask_i[2 * i],
             mask_start_bit);
    thread_grad_cmax += temp_cmax_g;
    clip_bwd(grad2_i[i].y, temp_cmax_g, grad2_i[i].y, cmask_i[2 * i],
             mask_start_bit);
    thread_grad_cmax += temp_cmax_g;
  }

  grad8[i] = grad8_i;
  if (grad_cmax) {
    if (threadIdx.x == 0) {
      block_grad_cmax = 0;
    }
    __syncthreads();

    if (thread_grad_cmax != 0) {
      atomicAdd(&block_grad_cmax, thread_grad_cmax);
    }
    __syncthreads();
    if (threadIdx.x == 0 && block_grad_cmax != 0) {
      atomicAdd(grad_cmax, __float2half(block_grad_cmax));
    }
  }
}

template <>
void launch_d_cmax<float>(float *grad_ptr, float *grad_cmax_ptr,
                          const uint8_t *clip_mask_ptr, int numel,
                          int mask_start_bit, cudaStream_t stream) {
  if (numel % 4 != 0) {
    throw std::runtime_error("violate numel % 4 = 0");
  }
  if (grad_cmax_ptr) {
    cudaMemsetAsync(grad_cmax_ptr, 0, sizeof(float), stream);
  }
  int ele_per_block = MAX_THREADS * 4;
  int grid_dim = numel / ele_per_block + 1;
  d_cmax_kernel<<<grid_dim + 1, MAX_THREADS, 0, stream>>>(
      grad_ptr, grad_cmax_ptr, clip_mask_ptr, numel, mask_start_bit);
}

template <>
void launch_d_cmax<__half>(__half *grad_ptr, __half *grad_cmax_ptr,
                           const uint8_t *clip_mask_ptr, int numel,
                           int mask_start_bit, cudaStream_t stream) {
  if (numel % 8 != 0) {
    throw std::runtime_error("violate numel % 8 = 0");
  }
  if (grad_cmax_ptr) {
    // zero_grad<<<1, 1>>>(grad_cmax_ptr);
    cudaMemsetAsync(grad_cmax_ptr, 0, sizeof(__half), stream);
  }
  int ele_per_block = MAX_THREADS * 8;
  int grid_dim = numel / ele_per_block + 1;
  d_cmax_kernel<<<grid_dim + 1, MAX_THREADS, 0, stream>>>(
      grad_ptr, grad_cmax_ptr, clip_mask_ptr, numel, mask_start_bit);
}

template <typename T>
__global__ void quantize_kernel(int8_t *q_ptr, uint8_t *clip_mask_ptr,
                                float *alpha_ptr, const T *f_ptr,
                                const T *clip_max_ptr, int batch_tokens,
                                int hidden_size, int mask_start_bit,
                                LSLayout out_layout) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 4 >= (batch_tokens * hidden_size)) return;

  float clip_max_val = clip_max_ptr[0];

  int32_t *q_weight4_ptr = reinterpret_cast<int32_t *>(q_ptr);
  const float4 *weight4_ptr = reinterpret_cast<const float4 *>(f_ptr);
  uint32_t *clip_mask4_ptr = reinterpret_cast<uint32_t *>(clip_mask_ptr);

  float4 weight4 = weight4_ptr[i];
  int8_t q_weight[4];
  uint8_t clip_mask[4];
  q_weight[0] = quantize(weight4.x, clip_max_val, clip_mask[0], mask_start_bit);
  q_weight[1] = quantize(weight4.y, clip_max_val, clip_mask[1], mask_start_bit);
  q_weight[2] = quantize(weight4.z, clip_max_val, clip_mask[2], mask_start_bit);
  q_weight[3] = quantize(weight4.w, clip_max_val, clip_mask[3], mask_start_bit);

  int output_index;
  if (out_layout != kRowMajor) {
    int row_id = (i * 4) / hidden_size;
    int col_id = (i * 4) % hidden_size;
    if (out_layout == kCol32) {
      output_index =
          row_major2flat_col32(row_id, col_id, batch_tokens, hidden_size) / 4;
    } else if (out_layout == kCOL4_4R2_8C) {
      output_index =
          row_major2cublaslt_weight(col_id, row_id, batch_tokens * 32, false) /
          4;
    } else if (out_layout == kCOL32_2R_4R4) {
      output_index =
          row_major2cublaslt_weight(col_id, row_id, batch_tokens * 32, true) /
          4;
    }

  } else {
    output_index = i;
  }

  q_weight4_ptr[output_index] = reinterpret_cast<int32_t *>(q_weight)[0];
  if (clip_mask_ptr) {
    clip_mask4_ptr[i] |= reinterpret_cast<uint32_t *>(clip_mask)[0];
  }

  if (i * 4 < batch_tokens && alpha_ptr) {
    float4 *alpha4_ptr = reinterpret_cast<float4 *>(alpha_ptr);
    float input_cmax = clip_max_val;
    float weight_cmax = clip_max_ptr[1];
    float output_cmax = clip_max_ptr[2];
    alpha_ptr[i] = input_cmax * weight_cmax / (output_cmax * kQuantRangeI8);
    float alpha = input_cmax * weight_cmax / (output_cmax * kQuantRangeI8);
    float4 alpha4;
    alpha4.x = alpha4.y = alpha4.z = alpha4.w = alpha;
    alpha4_ptr[i] = alpha4;
  }
}

template <>
__global__ void quantize_kernel<__half>(int8_t *q_ptr, uint8_t *clip_mask_ptr,
                                        float *alpha_ptr, const __half *f_ptr,
                                        const __half *clip_max_ptr,
                                        int batch_tokens, int hidden_size,
                                        int mask_start_bit,
                                        LSLayout out_layout) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 8 >= (batch_tokens * hidden_size)) return;

  float clip_max_val = __half2float(clip_max_ptr[0]);

  int64_t *q_weight8_ptr = reinterpret_cast<int64_t *>(q_ptr);
  const float4 *weight8_ptr = reinterpret_cast<const float4 *>(f_ptr);
  uint64_t *clip_mask8_ptr = reinterpret_cast<uint64_t *>(clip_mask_ptr);

  float4 weight8 = weight8_ptr[i];
  int8_t q_weight[8];
  uint8_t clip_mask[8];
  __half2 *weight2_ptr = reinterpret_cast<__half2 *>(&weight8);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    q_weight[i * 2] = quantize(__half2float(weight2_ptr[i].x), clip_max_val,
                               clip_mask[i * 2], mask_start_bit);
    q_weight[i * 2 + 1] = quantize(__half2float(weight2_ptr[i].y), clip_max_val,
                                   clip_mask[i * 2 + 1], mask_start_bit);
  }

  int output_index;
  if (out_layout != kRowMajor) {
    int row_id = (i * 8) / hidden_size;
    int col_id = (i * 8) % hidden_size;
    if (out_layout == kCol32) {
      output_index =
          row_major2flat_col32(row_id, col_id, batch_tokens, hidden_size) / 8;
    } else if (out_layout == kCOL4_4R2_8C) {
      output_index =
          row_major2cublaslt_weight(col_id, row_id, batch_tokens * 32, false) /
          8;
    } else if (out_layout == kCOL32_2R_4R4) {
      output_index =
          row_major2cublaslt_weight(col_id, row_id, batch_tokens * 32, true) /
          8;
    }

  } else {
    output_index = i;
  }
  q_weight8_ptr[output_index] = reinterpret_cast<int64_t *>(q_weight)[0];
  if (clip_mask_ptr) {
    clip_mask8_ptr[i] |= reinterpret_cast<uint64_t *>(clip_mask)[0];
  }
  if (i * 4 < batch_tokens && alpha_ptr) {
    float4 *alpha4_ptr = reinterpret_cast<float4 *>(alpha_ptr);
    float input_cmax = clip_max_val;
    float weight_cmax = __half2float(clip_max_ptr[1]);
    float output_cmax = __half2float(clip_max_ptr[2]);
    float alpha = input_cmax * weight_cmax / (output_cmax * kQuantRangeI8);
    float4 alpha4;
    alpha4.x = alpha4.y = alpha4.z = alpha4.w = alpha;
    alpha4_ptr[i] = alpha4;
  }
}

template <>
void launch_quantize<float>(int8_t *q_ptr, uint8_t *clip_mask_ptr,
                            float *alpha_ptr, const float *f_ptr,
                            const float *clip_max_ptr, int batch_tokens,
                            int hidden_size, int mask_start_bit,
                            cudaStream_t stream, LSLayout out_layout) {
  if ((batch_tokens * hidden_size) % 4 != 0) {
    throw std::runtime_error("violate (batch_tokens*hidden_size) % 4 = 0");
  }
  int ele_per_block = MAX_THREADS * 4;
  int grid_dim = (batch_tokens * hidden_size) / ele_per_block + 1;
  quantize_kernel<<<grid_dim + 1, MAX_THREADS, 0, stream>>>(
      q_ptr, clip_mask_ptr, alpha_ptr, f_ptr, clip_max_ptr, batch_tokens,
      hidden_size, mask_start_bit, out_layout);
}

template <>
void launch_quantize<__half>(int8_t *q_ptr, uint8_t *clip_mask_ptr,
                             float *alpha_ptr, const __half *f_ptr,
                             const __half *clip_max_ptr, int batch_tokens,
                             int hidden_size, int mask_start_bit,
                             cudaStream_t stream, LSLayout out_layout) {
  if ((batch_tokens * hidden_size) % 8 != 0) {
    throw std::runtime_error("violate (batch_tokens*hidden_size) % 8 = 0");
  }
  int ele_per_block = MAX_THREADS * 8;
  int grid_dim = (batch_tokens * hidden_size) / ele_per_block + 1;
  quantize_kernel<<<grid_dim + 1, MAX_THREADS, 0, stream>>>(
      q_ptr, clip_mask_ptr, alpha_ptr, f_ptr, clip_max_ptr, batch_tokens,
      hidden_size, mask_start_bit, out_layout);
}

__global__ void dequantize_kernel(float *f_ptr, const int8_t *q_ptr,
                                  const float *clip_max_ptr, int batch_tokens,
                                  int hidden_size, int mask_start_bit,
                                  bool in_col32) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 4 >= (batch_tokens * hidden_size)) return;

  float clip_max_val = clip_max_ptr[0];

  const int32_t *q_weight4_ptr = reinterpret_cast<const int32_t *>(q_ptr);
  float4 *weight4_ptr = reinterpret_cast<float4 *>(f_ptr);

  float4 weight4;

  int input_index;
  if (in_col32) {
    int row_id = (i * 4) / hidden_size;
    int col_id = (i * 4) % hidden_size;
    input_index =
        row_major2flat_col32(row_id, col_id, batch_tokens, hidden_size) / 4;
  } else {
    input_index = i;
  }

  int32_t q_weight_i32 = q_weight4_ptr[input_index];
  int8_t *q_weight = reinterpret_cast<int8_t *>(&q_weight_i32);
  weight4.x = dequantize(q_weight[0], clip_max_val);
  weight4.y = dequantize(q_weight[1], clip_max_val);
  weight4.z = dequantize(q_weight[2], clip_max_val);
  weight4.w = dequantize(q_weight[3], clip_max_val);

  weight4_ptr[i] = weight4;
}

__global__ void dequantize_kernel(__half *f_ptr, const int8_t *q_ptr,
                                  const __half *clip_max_ptr, int batch_tokens,
                                  int hidden_size, int mask_start_bit,
                                  bool in_col32) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 8 >= (batch_tokens * hidden_size)) return;

  float clip_max_val = __half2float(clip_max_ptr[0]);

  const int64_t *q_weight8_ptr = reinterpret_cast<const int64_t *>(q_ptr);
  float4 *weight8_ptr = reinterpret_cast<float4 *>(f_ptr);

  float4 weight8;

  int input_index;
  if (in_col32) {
    int row_id = (i * 8) / hidden_size;
    int col_id = (i * 8) % hidden_size;
    input_index =
        row_major2flat_col32(row_id, col_id, batch_tokens, hidden_size) / 8;
  } else {
    input_index = i;
  }
  int64_t q_weight_i64 = q_weight8_ptr[input_index];
  int8_t *q_weight = reinterpret_cast<int8_t *>(&q_weight_i64);
  __half2 *weight2_ptr = reinterpret_cast<__half2 *>(&weight8);
#pragma unroll
  for (int j = 0; j < 4; j++) {
    weight2_ptr[j].x = __float2half(dequantize(q_weight[j * 2], clip_max_val));
    weight2_ptr[j].y =
        __float2half(dequantize(q_weight[j * 2 + 1], clip_max_val));
  }

  weight8_ptr[i] = weight8;
}

template <>
void launch_dequantize<float>(float *f_ptr, const int8_t *q_ptr,
                              const float *clip_max_ptr, int batch_tokens,
                              int hidden_size, int mask_start_bit,
                              cudaStream_t stream, bool in_col32) {
  if ((batch_tokens * hidden_size) % 4 != 0) {
    throw std::runtime_error("violate (batch_tokens*hidden_size) % 4 = 0");
  }
  int ele_per_block = MAX_THREADS * 4;
  int grid_dim = (batch_tokens * hidden_size) / ele_per_block + 1;
  dequantize_kernel<<<grid_dim + 1, MAX_THREADS, 0, stream>>>(
      f_ptr, q_ptr, clip_max_ptr, batch_tokens, hidden_size, mask_start_bit,
      in_col32);
}

template <>
void launch_dequantize<__half>(__half *f_ptr, const int8_t *q_ptr,
                               const __half *clip_max_ptr, int batch_tokens,
                               int hidden_size, int mask_start_bit,
                               cudaStream_t stream, bool in_col32) {
  if ((batch_tokens * hidden_size) % 8 != 0) {
    throw std::runtime_error("violate (batch_tokens*hidden_size) % 8 = 0");
  }
  int ele_per_block = MAX_THREADS * 8;
  int grid_dim = (batch_tokens * hidden_size) / ele_per_block + 1;
  dequantize_kernel<<<grid_dim + 1, MAX_THREADS, 0, stream>>>(
      f_ptr, q_ptr, clip_max_ptr, batch_tokens, hidden_size, mask_start_bit,
      in_col32);
}
}  // namespace cuda
}  // namespace lightseq
