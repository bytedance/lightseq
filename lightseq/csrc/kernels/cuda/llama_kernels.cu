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
__global__ void kernel_llama_padding(const T* token_emb, const int* token_ids,
                                     T* output, T* pad_mask_ptr,
                                     int* left_pad_len_ptr, int batch_size,
                                     int beam_size, int seq_len, int hidden_dim,
                                     int padding_id, int max_step,
                                     int step_offset) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size * beam_size * seq_len * hidden_dim) {
    return;
  }
  int batch_idx, beam_idx, seq_idx, state_idx;
  decompose_4dim(idx, beam_size, seq_len, hidden_dim, &batch_idx, &beam_idx,
                 &seq_idx, &state_idx);
  int token_idx = flat_3dim(batch_idx, beam_idx, seq_idx + step_offset,
                            beam_size, max_step);
  int token_id = token_ids[token_idx];
  int batch_beam_idx = batch_idx * beam_size + beam_idx;

  float4& output_val = ((float4*)output)[idx];
  if (token_id == padding_id) {
    if (state_idx == 0) {
      pad_mask_ptr[token_idx] = CUDA_FLOAT_INF_NEG;
      atomicAdd(left_pad_len_ptr + batch_beam_idx, 1);
    }
    output_val.x = 0.;
    output_val.y = 0.;
    output_val.z = 0.;
    output_val.w = 0.;
  }
}

template <typename T>
__global__ void kernel_llama_embedding(const T* token_emb, const int* token_ids,
                                       T* output, T* pad_mask_ptr,
                                       int* left_pad_len_ptr, int batch_size,
                                       int beam_size, int seq_len,
                                       int hidden_dim, int padding_id,
                                       int max_step, int step_offset) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size * beam_size * seq_len * hidden_dim) {
    return;
  }
  int batch_idx, beam_idx, seq_idx, state_idx;
  decompose_4dim(idx, beam_size, seq_len, hidden_dim, &batch_idx, &beam_idx,
                 &seq_idx, &state_idx);
  int token_idx = flat_3dim(batch_idx, beam_idx, seq_idx + step_offset,
                            beam_size, max_step);
  int token_id = token_ids[token_idx];

  float4& output_val = ((float4*)output)[idx];
  if (token_id != padding_id) {
    if (state_idx == 0) {
      pad_mask_ptr[token_idx] = 0;
    }
    output_val = ((float4*)token_emb)[token_id * hidden_dim + state_idx];
  }
}

template <>
__global__ void kernel_llama_padding<__half>(
    const __half* token_emb, const int* token_ids, __half* output,
    __half* pad_mask_ptr, int* left_pad_len_ptr, int batch_size, int beam_size,
    int seq_len, int hidden_dim, int padding_id, int max_step,
    int step_offset) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size * beam_size * seq_len * hidden_dim) {
    return;
  }
  int batch_idx, beam_idx, seq_idx, state_idx;
  decompose_4dim(idx, beam_size, seq_len, hidden_dim, &batch_idx, &beam_idx,
                 &seq_idx, &state_idx);
  int token_idx = flat_3dim(batch_idx, beam_idx, seq_idx + step_offset,
                            beam_size, max_step);
  int token_id = token_ids[token_idx];
  int batch_beam_idx = batch_idx * beam_size + beam_idx;

  float4& output_val = ((float4*)output)[idx];
  if (token_id == padding_id) {
    if (state_idx == 0) {
      pad_mask_ptr[token_idx] = __float2half(CUDA_FLOAT_INF_NEG);
      atomicAdd(left_pad_len_ptr + batch_beam_idx, 1);
    }
    output_val.x = 0.f;
    output_val.y = 0.f;
    output_val.z = 0.f;
    output_val.w = 0.f;
  }
}

template <>
__global__ void kernel_llama_embedding<__half>(
    const __half* token_emb, const int* token_ids, __half* output,
    __half* pad_mask_ptr, int* left_pad_len_ptr, int batch_size, int beam_size,
    int seq_len, int hidden_dim, int padding_id, int max_step,
    int step_offset) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size * beam_size * seq_len * hidden_dim) {
    return;
  }
  int batch_idx, beam_idx, seq_idx, state_idx;
  decompose_4dim(idx, beam_size, seq_len, hidden_dim, &batch_idx, &beam_idx,
                 &seq_idx, &state_idx);
  int token_idx = flat_3dim(batch_idx, beam_idx, seq_idx + step_offset,
                            beam_size, max_step);
  int token_id = token_ids[token_idx];

  float4& output_val = ((float4*)output)[idx];

  if (token_id != padding_id) {
    if (state_idx == 0) {
      pad_mask_ptr[token_idx] = __float2half(0.f);
    }
    output_val = ((float4*)token_emb)[token_id * hidden_dim + state_idx];
  }
}

template <>
void launch_llama_embedding<float>(const float* token_emb, const int* tokens,
                                   float* output, float* pad_mask_ptr,
                                   int* left_pad_len_ptr, int batch_size,
                                   int beam_size, int hidden_dim,
                                   int step_offset, int seq_len, int max_step,
                                   int padding_id, cudaStream_t stream) {
  if (seq_len + step_offset >= max_step) {
    throw std::runtime_error("violate seq_len + step_offset < max_step");
  }
  if (hidden_dim % 4) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  hidden_dim >>= 2;
  int nele = (batch_size * beam_size * seq_len * hidden_dim);
  int nblock = (nele + MAX_THREADS - 1) / MAX_THREADS;
  kernel_llama_padding<float><<<nblock, MAX_THREADS, 0, stream>>>(
      token_emb, tokens, output, pad_mask_ptr, left_pad_len_ptr, batch_size,
      beam_size, seq_len, hidden_dim, padding_id, max_step, step_offset);

  kernel_llama_embedding<float><<<nblock, MAX_THREADS, 0, stream>>>(
      token_emb, tokens, output, pad_mask_ptr, left_pad_len_ptr, batch_size,
      beam_size, seq_len, hidden_dim, padding_id, max_step, step_offset);
}

template <>
void launch_llama_embedding<__half>(const __half* token_emb, const int* tokens,
                                    __half* output, __half* pad_mask_ptr,
                                    int* left_pad_len_ptr, int batch_size,
                                    int beam_size, int hidden_dim,
                                    int step_offset, int seq_len, int max_step,
                                    int padding_id, cudaStream_t stream) {
  if (seq_len + step_offset >= max_step) {
    throw std::runtime_error("violate seq_len + step_offset < max_step");
  }
  if (hidden_dim % 8) {
    throw std::runtime_error("violate hidden_dim % 8 = 0");
  }
  hidden_dim >>= 3;
  int nele = (batch_size * beam_size * seq_len * hidden_dim);
  int nblock = (nele + MAX_THREADS - 1) / MAX_THREADS;
  kernel_llama_padding<__half><<<nblock, MAX_THREADS, 0, stream>>>(
      token_emb, tokens, output, pad_mask_ptr, left_pad_len_ptr, batch_size,
      beam_size, seq_len, hidden_dim, padding_id, max_step, step_offset);
  kernel_llama_embedding<__half><<<nblock, MAX_THREADS, 0, stream>>>(
      token_emb, tokens, output, pad_mask_ptr, left_pad_len_ptr, batch_size,
      beam_size, seq_len, hidden_dim, padding_id, max_step, step_offset);
}

template void launch_llama_embedding<float>(
    const float* token_emb, const int* tokens, float* output,
    float* pad_mask_ptr, int* left_pad_len_ptr, int batch_size, int beam_size,
    int hidden_dim, int step_offset, int seq_len, int max_step, int padding_id,
    cudaStream_t stream);

template void launch_llama_embedding<__half>(
    const __half* token_emb, const int* tokens, __half* output,
    __half* pad_mask_ptr, int* left_pad_len_ptr, int batch_size, int beam_size,
    int hidden_dim, int step_offset, int seq_len, int max_step, int padding_id,
    cudaStream_t stream);

template <typename T>
__global__ void kernel_split_rotary_position_qkv(
    const T* input_ptr, const T* sin_ptr, const T* cos_ptr, T* q_out,
    T* cache_k_out, T* cache_v_out, size_t batch_size, size_t max_step,
    size_t nhead, size_t offset_seq_len, size_t query_len, size_t head_dim,
    size_t max_thread_num) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= max_thread_num) {
    return;
  }
  int batch_idx, qkv_idx, head_idx, seq_idx, head_dim_idx;
  decompose_5dim(idx, query_len, 3, nhead, head_dim, &batch_idx, &seq_idx,
                 &qkv_idx, &head_idx, &head_dim_idx);

  size_t output_idx = 0;
  if (qkv_idx) {
    output_idx = flat_4dim(batch_idx, head_idx, offset_seq_len + seq_idx,
                           head_dim_idx, nhead, max_step, head_dim);
  } else {
    output_idx = flat_4dim(batch_idx, head_idx, seq_idx, head_dim_idx, nhead,
                           query_len, head_dim);
  }

  // cos part
  T state_val1 = *(input_ptr + idx);

  if (qkv_idx == 2) {
    *(cache_v_out + output_idx) = state_val1;
  } else {
    T cos_val = *(cos_ptr + (offset_seq_len + seq_idx) * head_dim / 2 +
                  (head_dim_idx % (head_dim / 2)));
    T sin_val = *(sin_ptr + (offset_seq_len + seq_idx) * head_dim / 2 +
                  (head_dim_idx % (head_dim / 2)));
    T out_val = 0.;
    if (head_dim_idx < head_dim / 2) {
      T state_val2 = *(input_ptr + idx + head_dim / 2);
      out_val = state_val1 * cos_val - state_val2 * sin_val;
    } else {
      T state_val2 = *(input_ptr + idx - head_dim / 2);
      out_val = state_val1 * cos_val + state_val2 * sin_val;
    }

    if (qkv_idx == 0) {
      *(q_out + output_idx) = out_val;
    } else {
      *(cache_k_out + output_idx) = out_val;
    }
  }
}

template <typename T>
void launch_split_rotary_position_qkv(const T* input_ptr, const T* sin_ptr,
                                      const T* cos_ptr, T* q_out,
                                      T* cache_k_out, T* cache_v_out,
                                      size_t max_step, size_t batch_size,
                                      size_t nhead, size_t offset_seq_len,
                                      size_t query_len, size_t head_dim,
                                      cudaStream_t stream) {
  size_t nele = 3 * batch_size * nhead * query_len * head_dim;
  size_t nblock = (nele + MAX_THREADS - 1) / MAX_THREADS;
  kernel_split_rotary_position_qkv<T><<<nblock, MAX_THREADS, 0, stream>>>(
      input_ptr, sin_ptr, cos_ptr, q_out, cache_k_out, cache_v_out, batch_size,
      max_step, nhead, offset_seq_len, query_len, head_dim, nele);
}

template void launch_split_rotary_position_qkv<float>(
    const float* input_ptr, const float* sin_ptr, const float* cos_ptr,
    float* q_out, float* cache_k_out, float* cache_v_out, size_t max_step,
    size_t batch_size, size_t nhead, size_t offset_seq_len, size_t query_len,
    size_t head_dim, cudaStream_t stream);

template void launch_split_rotary_position_qkv<__half>(
    const __half* input_ptr, const __half* sin_ptr, const __half* cos_ptr,
    __half* q_out, __half* cache_k_out, __half* cache_v_out, size_t max_step,
    size_t batch_size, size_t nhead, size_t offset_seq_len, size_t query_len,
    size_t head_dim, cudaStream_t stream);

template <typename T>
__global__ void kernel_silu_elewise_product(const T* inp_ptr, T* out_ptr,
                                            size_t seq_len, size_t inner_size,
                                            size_t max_thread_num) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= max_thread_num) {
    return;
  }
  int inpA_idx = idx / inner_size * inner_size * 2 + idx % inner_size;
  int inpB_idx = inpA_idx + inner_size;
  const T& inpA = *(inp_ptr + inpA_idx);
  const T& inpB = *(inp_ptr + inpB_idx);
  *(out_ptr + idx) = inpA / (1.f + __expf(-inpA)) * inpB;
}

template <>
__global__ void kernel_silu_elewise_product<__half>(const __half* inp_ptr,
                                                    __half* out_ptr,
                                                    size_t seq_len,
                                                    size_t inner_size,
                                                    size_t max_thread_num) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= max_thread_num) {
    return;
  }
  // const __half& ele_product = *(inpA_ptr + idx);
  int inpA_idx = idx / inner_size * inner_size * 2 + idx % inner_size;
  int inpB_idx = inpA_idx + inner_size;
  const __half& inpA = *(inp_ptr + inpA_idx);
  const __half& inpB = *(inp_ptr + inpB_idx);
  *(out_ptr + idx) = inpA / __float2half(1.f + __expf(-inpA)) * inpB;
}

template <typename T>
void launch_silu_elewise_product(const T* inp_ptr, T* out_ptr,
                                 size_t batch_size, size_t seq_len,
                                 size_t inner_size, cudaStream_t stream) {
  size_t nele = batch_size * seq_len * inner_size;
  size_t nblock = (nele + MAX_THREADS - 1) / MAX_THREADS;
  kernel_silu_elewise_product<T><<<nblock, MAX_THREADS, 0, stream>>>(
      inp_ptr, out_ptr, seq_len, inner_size, nele);
}

template void launch_silu_elewise_product<float>(
    const float* inp_ptr, float* out_ptr, size_t batch_size, size_t seq_len,
    size_t inner_size, cudaStream_t stream);
template void launch_silu_elewise_product<__half>(
    const __half* inp_ptr, __half* out_ptr, size_t batch_size, size_t seq_len,
    size_t inner_size, cudaStream_t stream);

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
__global__ void ker_rms_layer_norm_with_res(const T* inp_ptr,
                                            const T* scale_ptr, T* out_ptr,
                                            T* res_ptr, T* rms_ptr,
                                            size_t hidden_dim,
                                            const float ln_epsilon) {
  // step 0. compute local sum
  float l_square_sum = 0;
  const T* thread_inp = inp_ptr + blockIdx.x * hidden_dim;
  T* res_thread_out = res_ptr + blockIdx.x * hidden_dim;
  for (uint idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
    l_square_sum += thread_inp[idx] * thread_inp[idx];
    res_thread_out[idx] = thread_inp[idx];
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
__global__ void ker_rms_layer_norm_with_res<__half>(
    const __half* inp_ptr, const __half* scale_ptr, __half* out_ptr,
    __half* res_ptr, __half* rms_ptr, size_t hidden_dim,
    const float ln_epsilon) {
  // step 0. compute local sum
  float l_square_sum = 0;
  const __half* thread_inp = inp_ptr + blockIdx.x * hidden_dim;
  __half* res_thread_out = res_ptr + blockIdx.x * hidden_dim;
  for (uint idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
    float float_inp = __half2float(thread_inp[idx]);
    l_square_sum += float_inp * float_inp;
    res_thread_out[idx] = thread_inp[idx];
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
                           T* res_ptr, T* rms_ptr, size_t batch_tokens,
                           size_t hidden_dim, cudaStream_t stream,
                           const float ln_epsilon) {
  int nthread = std::min(((hidden_dim + 31) / 32) * 32, size_t(MAX_THREADS));
  dim3 grid_dim(batch_tokens);
  dim3 block_dim(nthread);

  if (res_ptr == nullptr) {
    ker_rms_layer_norm<T><<<grid_dim, block_dim, 0, stream>>>(
        inp_ptr, scale_ptr, out_ptr, rms_ptr, hidden_dim, ln_epsilon);
  } else {
    ker_rms_layer_norm_with_res<T><<<grid_dim, block_dim, 0, stream>>>(
        inp_ptr, scale_ptr, out_ptr, res_ptr, rms_ptr, hidden_dim, ln_epsilon);
  }
}

template void launch_rms_layer_norm<float>(
    const float* inp_ptr, const float* scale_ptr, float* out_ptr,
    float* res_ptr, float* rms_ptr, size_t batch_tokens, size_t hidden_dim,
    cudaStream_t stream, const float ln_epsilon);
template void launch_rms_layer_norm<__half>(
    const __half* inp_ptr, const __half* scale_ptr, __half* out_ptr,
    __half* res_ptr, __half* rms_ptr, size_t batch_tokens, size_t hidden_dim,
    cudaStream_t stream, const float ln_epsilon);

}  // namespace cuda
}  // namespace lightseq
