#include "kernels.h"
#include "llama_kernels.h"
#include "common.h"
#include <cub/cub.cuh>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace lightseq {
namespace cuda {

template <typename T>
__global__ void kernel_rotary_position_qk(const T* input_ptr, const T* sin_ptr,
                                          const T* cos_ptr, T* output_ptr,
                                          size_t max_step, size_t nhead,
                                          size_t offset_seq_len,
                                          size_t query_len, size_t head_dim,
                                          size_t max_thread_num) {
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
  if (head_dim_idx < head_dim / 2) {
    T state_val2 = *(input_ptr + idx + head_dim / 2);
    *(output_ptr + idx) = state_val1 * cos_val - state_val2 * sin_val;
  } else {
    T state_val2 = *(input_ptr + idx - head_dim / 2);
    *(output_ptr + idx) = state_val1 * cos_val + state_val2 * sin_val;
  }
}

template <typename T>
void launch_rotary_position_qk(const T* input_ptr, const T* sin_ptr,
                               const T* cos_ptr, T* output_ptr, size_t max_step,
                               size_t batch_size, size_t nhead,
                               size_t offset_seq_len, size_t query_len,
                               size_t head_dim, cudaStream_t stream) {
  size_t nele = batch_size * nhead * query_len * head_dim;
  size_t nblock = (nele + MAX_THREADS - 1) / MAX_THREADS;
  kernel_rotary_position_qk<T><<<nblock, MAX_THREADS, 0, stream>>>(
      input_ptr, sin_ptr, cos_ptr, output_ptr, max_step, nhead, offset_seq_len,
      query_len, head_dim, nele);
}

template void launch_rotary_position_qk<float>(
    const float* input_ptr, const float* sin_ptr, const float* cos_ptr,
    float* output_ptr, size_t max_step, size_t batch_size, size_t nhead,
    size_t offset_seq_len, size_t query_len, size_t head_dim,
    cudaStream_t stream);

template void launch_rotary_position_qk<__half>(
    const __half* input_ptr, const __half* sin_ptr, const __half* cos_ptr,
    __half* output_ptr, size_t max_step, size_t batch_size, size_t nhead,
    size_t offset_seq_len, size_t query_len, size_t head_dim,
    cudaStream_t stream);

template <typename T>
__global__ void kernel_elewise_product_silu(const T* inpA_ptr,
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
__global__ void kernel_elewise_product_silu<__half>(
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
void launch_elewise_product_silu(const T* inpA_ptr, const T* inpB_ptr,
                                 T* out_ptr, size_t batch_size, size_t seq_len,
                                 size_t inner_size, cudaStream_t stream) {
  size_t nele = batch_size * seq_len * inner_size;
  size_t nblock = (nele + MAX_THREADS - 1) / MAX_THREADS;
  kernel_elewise_product_silu<T><<<nblock, MAX_THREADS, 0, stream>>>(
      inpA_ptr, inpB_ptr, out_ptr, seq_len, inner_size, nele);
}

template void launch_elewise_product_silu<float>(
    const float* inpA_ptr, const float* inpB_ptr, float* out_ptr,
    size_t batch_size, size_t seq_len, size_t inner_size, cudaStream_t stream);
template void launch_elewise_product_silu<__half>(
    const __half* inpA_ptr, const __half* inpB_ptr, __half* out_ptr,
    size_t batch_size, size_t seq_len, size_t inner_size, cudaStream_t stream);
}  // namespace cuda
}  // namespace lightseq
