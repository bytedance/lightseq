#include "kernels.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

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

  float clip_max_val = clip_max_ptr[1];

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
  clip_mask4_ptr[i] = reinterpret_cast<uint32_t *>(clip_mask)[0];

  if (blockIdx.x == 0 && threadIdx.x == 0 && alpha_ptr) {
    float input_cmax = clip_max_ptr[0];
    float weight_cmax = clip_max_val;
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

  float clip_max_val = __half2float(clip_max_ptr[1]);

  int64_t *q_weight8_ptr = reinterpret_cast<int64_t *>(q_ptr);
  const float4 *weight8_ptr = reinterpret_cast<const float4 *>(f_ptr);
  uint64_t *clip_mask8_ptr = reinterpret_cast<uint64_t *>(clip_mask_ptr);

  float4 weight8 = weight8_ptr[i];
  int8_t q_weight[8];
  uint8_t clip_mask[8];
  __half2 *weight2_ptr = reinterpret_cast<__half2 *>(&weight8);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    q_weight[i * 2] = quantize(weight2_ptr[i].x, clip_max_val, clip_mask[i * 2],
                               mask_start_bit);
    q_weight[i * 2 + 1] = quantize(weight2_ptr[i].y, clip_max_val,
                                   clip_mask[i * 2 + 1], mask_start_bit);
  }

  q_weight8_ptr[i] = reinterpret_cast<int64_t *>(q_weight)[0];
  clip_mask8_ptr[i] = reinterpret_cast<uint64_t *>(clip_mask)[0];

  if (blockIdx.x == 0 && threadIdx.x == 0 && alpha_ptr) {
    float input_cmax = __half2float(clip_max_ptr[0]);
    float weight_cmax = clip_max_val;
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
  int grid_dim = numel / ele_per_block;
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
  int grid_dim = numel / ele_per_block;
  quantize_kernel<<<grid_dim + 1, MAX_THREADS, 0, stream>>>(
      q_ptr, clip_mask_ptr, alpha_ptr, f_ptr, clip_max_ptr, numel,
      mask_start_bit);
}

template <typename T>
__global__ void fake_quantize_kernel(uint8_t *clip_mask_ptr, float *alpha_ptr,
                                     T *f_ptr, const T *clip_max_ptr, int numel,
                                     int mask_start_bit);
template <>
__global__ void fake_quantize_kernel<float>(uint8_t *clip_mask_ptr,
                                            float *alpha_ptr, float *f_ptr,
                                            const float *clip_max_ptr,
                                            int numel, int mask_start_bit) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 4 >= numel) return;

  float clip_max_val = clip_max_ptr[1];

  float4 *weight4_ptr = reinterpret_cast<float4 *>(f_ptr);
  uint32_t *clip_mask4_ptr;
  if (clip_mask_ptr) {
    clip_mask4_ptr = reinterpret_cast<uint32_t *>(clip_mask_ptr);
  }

  float4 weight4 = weight4_ptr[i];

  uint8_t clip_mask[4];

  weight4.x =
      fake_quantize(weight4.x, clip_max_val, clip_mask[0], mask_start_bit);
  weight4.y =
      fake_quantize(weight4.y, clip_max_val, clip_mask[1], mask_start_bit);
  weight4.z =
      fake_quantize(weight4.z, clip_max_val, clip_mask[2], mask_start_bit);
  weight4.w =
      fake_quantize(weight4.w, clip_max_val, clip_mask[3], mask_start_bit);
  weight4_ptr[i] = weight4;

  if (clip_mask_ptr) {
    clip_mask4_ptr[i] = reinterpret_cast<uint32_t *>(clip_mask)[0];
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
                                             float *alpha_ptr, __half *f_ptr,
                                             const __half *clip_max_ptr,
                                             int numel, int mask_start_bit) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 8 >= numel) return;

  float clip_max_val = __half2float(clip_max_ptr[1]);

  float4 *weight8_ptr = reinterpret_cast<float4 *>(f_ptr);
  uint64_t *clip_mask8_ptr;
  if (clip_mask_ptr) {
    clip_mask8_ptr = reinterpret_cast<uint64_t *>(clip_mask_ptr);
  }
  float4 weight8 = weight8_ptr[i];

  uint8_t clip_mask[8];
  __half2 *weight2_ptr = reinterpret_cast<__half2 *>(&weight8);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    weight2_ptr[i].x = fake_quantize(weight2_ptr[i].x, clip_max_val,
                                     clip_mask[i * 2], mask_start_bit);
    weight2_ptr[i].y = fake_quantize(weight2_ptr[i].y, clip_max_val,
                                     clip_mask[i * 2 + 1], mask_start_bit);
  }

  weight8_ptr[i] = weight8;
  if (clip_mask_ptr) {
    clip_mask8_ptr[i] = reinterpret_cast<uint64_t *>(clip_mask)[0];
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
                                 float *f_ptr, const float *clip_max_ptr,
                                 int numel, int mask_start_bit,
                                 cudaStream_t stream) {
  if (numel % 4 != 0) {
    throw std::runtime_error("violate numel % 4 = 0");
  }
  int ele_per_block = MAX_THREADS * 4;
  int grid_dim = numel / ele_per_block;
  fake_quantize_kernel<<<grid_dim + 1, MAX_THREADS, 0, stream>>>(
      clip_mask_ptr, alpha_ptr, f_ptr, clip_max_ptr, numel, mask_start_bit);
}

template <>
void launch_fake_quantize<__half>(uint8_t *clip_mask_ptr, float *alpha_ptr,
                                  __half *f_ptr, const __half *clip_max_ptr,
                                  int numel, int mask_start_bit,
                                  cudaStream_t stream) {
  if (numel % 8 != 0) {
    throw std::runtime_error("violate numel % 8 = 0");
  }
  int ele_per_block = MAX_THREADS * 8;
  int grid_dim = numel / ele_per_block;
  fake_quantize_kernel<<<grid_dim + 1, MAX_THREADS, 0, stream>>>(
      clip_mask_ptr, alpha_ptr, f_ptr, clip_max_ptr, numel, mask_start_bit);
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
  int grid_dim = numel / ele_per_block;
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
  int grid_dim = numel / ele_per_block;
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
  int grid_dim = numel / ele_per_block;
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
  int grid_dim = numel / ele_per_block;
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
  if (thread_grad_cmax != 0) {
    atomicAdd(&block_grad_cmax, thread_grad_cmax);
  }
  __syncthreads();
  if (threadIdx.x == 0 && block_grad_cmax != 0) {
    atomicAdd(grad_cmax, block_grad_cmax);
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

  if (thread_grad_cmax != 0) {
    atomicAdd(&block_grad_cmax, thread_grad_cmax);
  }
  __syncthreads();
  if (threadIdx.x == 0 && block_grad_cmax != 0) {
    atomicAdd(grad_cmax, __float2half(block_grad_cmax));
  }
}

template <>
void launch_d_cmax<float>(float *grad_ptr, float *grad_cmax_ptr,
                          const uint8_t *clip_mask_ptr, int numel,
                          int mask_start_bit, cudaStream_t stream) {
  if (numel % 4 != 0) {
    throw std::runtime_error("violate numel % 4 = 0");
  }
  zero_grad<<<1, 1>>>(grad_cmax_ptr);
  int ele_per_block = MAX_THREADS * 4;
  int grid_dim = numel / ele_per_block;
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
  zero_grad<<<1, 1>>>(grad_cmax_ptr);
  int ele_per_block = MAX_THREADS * 8;
  int grid_dim = numel / ele_per_block;
  d_cmax_kernel<<<grid_dim + 1, MAX_THREADS, 0, stream>>>(
      grad_ptr, grad_cmax_ptr, clip_mask_ptr, numel, mask_start_bit);
}
