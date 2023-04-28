/*
  Copyright 2023 Bytedance Lab-nlp
*/

#include "kernels.h"
#include "llama_kernels.h"
#include "common.h"
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include "block_reduce.h"
#include <cooperative_groups.h>
#include <cstddef>

namespace cg = cooperative_groups;

namespace lightseq {
namespace cuda {

template <typename T>
__global__ void kernel_rotary_position_qk(
    const T* input_ptr, const T* sin_ptr, const T* cos_ptr, T* output_ptr,
    size_t max_step, size_t nhead, size_t offset_seq_len, size_t query_len,
    size_t head_dim, size_t max_thread_num, int append_cache) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= max_thread_num) {
    return;
  }
  int batch_idx, head_idx, seq_idx, head_dim_idx;
  decompose_4dim(idx, nhead, query_len, head_dim, &batch_idx, &head_idx,
                 &seq_idx, &head_dim_idx);
  // cos part
  T state_val1 = *(input_ptr + idx);
  T cos_val = *(cos_ptr + (offset_seq_len + seq_idx) * head_dim / 2 +
                (head_dim_idx % (head_dim / 2)));
  T sin_val = *(sin_ptr + (offset_seq_len + seq_idx) * head_dim / 2 +
                (head_dim_idx % (head_dim / 2)));
  size_t output_idx =
      flat_4dim(batch_idx, head_idx, append_cache * offset_seq_len + seq_idx,
                head_dim_idx, nhead, max_step, head_dim);
  if (head_dim_idx < head_dim / 2) {
    T state_val2 = *(input_ptr + idx + head_dim / 2);
    *(output_ptr + output_idx) = state_val1 * cos_val - state_val2 * sin_val;
  } else {
    T state_val2 = *(input_ptr + idx - head_dim / 2);
    *(output_ptr + output_idx) = state_val1 * cos_val + state_val2 * sin_val;
  }
}

template <typename T>
void launch_rotary_position_qk(const T* input_ptr, const T* sin_ptr,
                               const T* cos_ptr, T* output_ptr, size_t max_step,
                               size_t batch_size, size_t nhead,
                               size_t offset_seq_len, size_t query_len,
                               size_t head_dim, bool append_cache,
                               cudaStream_t stream) {
  size_t nele = batch_size * nhead * query_len * head_dim;
  size_t nblock = (nele + MAX_THREADS - 1) / MAX_THREADS;
  kernel_rotary_position_qk<T><<<nblock, MAX_THREADS, 0, stream>>>(
      input_ptr, sin_ptr, cos_ptr, output_ptr, max_step, nhead, offset_seq_len,
      query_len, head_dim, nele, append_cache);
}

template void launch_rotary_position_qk<float>(
    const float* input_ptr, const float* sin_ptr, const float* cos_ptr,
    float* output_ptr, size_t max_step, size_t batch_size, size_t nhead,
    size_t offset_seq_len, size_t query_len, size_t head_dim, bool append_cache,
    cudaStream_t stream);

template void launch_rotary_position_qk<__half>(
    const __half* input_ptr, const __half* sin_ptr, const __half* cos_ptr,
    __half* output_ptr, size_t max_step, size_t batch_size, size_t nhead,
    size_t offset_seq_len, size_t query_len, size_t head_dim, bool append_cache,
    cudaStream_t stream);

template <typename T>
__global__ void kernel_silu_elewise_product(const T* inpA_ptr,
                                            const T* inpB_ptr, T* out_ptr,
                                            size_t seq_len, size_t inner_size,
                                            size_t max_thread_num) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= max_thread_num) {
    return;
  }
  int batch_idx, seq_idx, dim_idx;
  decompose_3dim(idx, seq_len, inner_size, &batch_idx, &seq_idx, &dim_idx);
  T ele_product = *(inpA_ptr + idx);
  *(out_ptr + idx) =
      ele_product / (1.f + expf(-ele_product)) * (*(inpB_ptr + idx));
}

template <>
__global__ void kernel_silu_elewise_product<__half>(
    const __half* inpA_ptr, const __half* inpB_ptr, __half* out_ptr,
    size_t seq_len, size_t inner_size, size_t max_thread_num) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= max_thread_num) {
    return;
  }
  int batch_idx, seq_idx, dim_idx;
  decompose_3dim(idx, seq_len, inner_size, &batch_idx, &seq_idx, &dim_idx);
  const __half& ele_product = *(inpA_ptr + idx);
  *(out_ptr + idx) = ele_product / __float2half(1.f + expf(-ele_product)) *
                     (*(inpB_ptr + idx));
}

template <typename T>
void launch_silu_elewise_product(const T* inpA_ptr, const T* inpB_ptr,
                                 T* out_ptr, size_t batch_size, size_t seq_len,
                                 size_t inner_size, cudaStream_t stream) {
  size_t nele = batch_size * seq_len * inner_size;
  size_t nblock = (nele + MAX_THREADS - 1) / MAX_THREADS;
  kernel_silu_elewise_product<T><<<nblock, MAX_THREADS, 0, stream>>>(
      inpA_ptr, inpB_ptr, out_ptr, seq_len, inner_size, nele);
}

template void launch_silu_elewise_product<float>(
    const float* inpA_ptr, const float* inpB_ptr, float* out_ptr,
    size_t batch_size, size_t seq_len, size_t inner_size, cudaStream_t stream);
template void launch_silu_elewise_product<__half>(
    const __half* inpA_ptr, const __half* inpB_ptr, __half* out_ptr,
    size_t batch_size, size_t seq_len, size_t inner_size, cudaStream_t stream);

template <typename T>
__global__ void ker_rms_layer_norm(const T* inp_ptr, const T* scale_ptr,
                                   T* out_ptr, T* rms_ptr, size_t hidden_dim,
                                   const float ln_epsilon) {
  // step 0. compute local sum
  float l_square_sum = 0;
  const T* thread_inp = inp_ptr + blockIdx.x * hidden_dim;
  for (uint idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
    l_square_sum += thread_inp[idx] * thread_inp[idx];
  }

  // step 1. compute reduce sum
  float mean_dim = float(hidden_dim);
  float kReduce[1] = {l_square_sum};
  blockReduce<ReduceType::kSum, 1>(kReduce);
  __shared__ float s_var;
  if (threadIdx.x == 0) {
    s_var = rsqrtf(kReduce[0] / mean_dim + ln_epsilon);
    rms_ptr[blockIdx.x] = s_var;
  }
  __syncthreads();

  // step 2. layer norm result
  T* thread_out = out_ptr + blockIdx.x * hidden_dim;
  for (uint idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
    thread_out[idx] = thread_inp[idx] * scale_ptr[idx] * s_var;
  }
}

template <>
__global__ void ker_rms_layer_norm<__half>(const __half* inp_ptr,
                                           const __half* scale_ptr,
                                           __half* out_ptr, __half* rms_ptr,
                                           size_t hidden_dim,
                                           const float ln_epsilon) {
  // step 0. compute local sum
  float l_square_sum = 0;
  const __half* thread_inp = inp_ptr + blockIdx.x * hidden_dim;
  for (uint idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
    float float_inp = __half2float(thread_inp[idx]);
    l_square_sum += float_inp * float_inp;
  }

  // step 1. compute reduce sum
  float mean_dim = float(hidden_dim);
  float kReduce[1] = {l_square_sum};
  blockReduce<ReduceType::kSum, 1>(kReduce);
  __shared__ __half s_var;
  if (threadIdx.x == 0) {
    s_var = __float2half(rsqrtf(kReduce[0] / mean_dim + ln_epsilon));
    if (rms_ptr != nullptr) rms_ptr[blockIdx.x] = s_var;
  }
  __syncthreads();

  // step 2. layer norm result
  __half* thread_out = out_ptr + blockIdx.x * hidden_dim;
  for (uint idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
    thread_out[idx] = thread_inp[idx] * scale_ptr[idx] * s_var;
  }
}

template <typename T>
void launch_rms_layer_norm(const T* inp_ptr, const T* scale_ptr, T* out_ptr,
                           T* rms_ptr, size_t batch_tokens, size_t hidden_dim,
                           cudaStream_t stream, const float ln_epsilon) {
  int nthread = std::min(((hidden_dim + 31) / 32) * 32, size_t(MAX_THREADS));
  dim3 grid_dim(batch_tokens);
  dim3 block_dim(nthread);

  ker_rms_layer_norm<T><<<grid_dim, block_dim, 0, stream>>>(
      inp_ptr, scale_ptr, out_ptr, rms_ptr, hidden_dim, ln_epsilon);
}

template void launch_rms_layer_norm<float>(
    const float* inp_ptr, const float* scale_ptr, float* out_ptr,
    float* rms_ptr, size_t batch_tokens, size_t hidden_dim, cudaStream_t stream,
    const float ln_epsilon);
template void launch_rms_layer_norm<__half>(
    const __half* inp_ptr, const __half* scale_ptr, __half* out_ptr,
    __half* rms_ptr, size_t batch_tokens, size_t hidden_dim,
    cudaStream_t stream, const float ln_epsilon);

}  // namespace cuda
}  // namespace lightseq
