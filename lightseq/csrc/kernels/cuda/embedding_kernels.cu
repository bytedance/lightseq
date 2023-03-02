#include <chrono>
#include <ctime>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "kernels.h"

#include "cuda_util.h"

namespace cg = cooperative_groups;
namespace lightseq {
namespace cuda {
/**
@brief: get_tokens_position
get tokens position in sequences that the padding tokens are ignored.

@thread
gridDim.x = batch_size
gridDim.y = 1
blockDim.x = min(seq_len, MAX_THREADS)
blockDim.y = 1

@param
output: [batch_size, seq_len, 2]
input: [batch_size, seq_len]
batch_size: the size of the current batch
seq_len: the sequence length of the current batch
padding_idx: padding index of the sentences (default: 2)
*/
__global__ void get_tokens_position(int *output, const int *input,
                                    int batch_size, int seq_len,
                                    int padding_idx) {
  int batch_id = blockIdx.x;
  int start_seq_id = threadIdx.x;
  int threads = blockDim.x;

  int batch_offset = batch_id * seq_len;
  int temp_offset[2];
  temp_offset[0] = 0;
  temp_offset[1] = batch_size * seq_len;

  int *temp = output;
  int pout = 0, pin = 1;
  int pout_idx, pin_idx, target_pos;

  for (int seq_id = start_seq_id; seq_id < seq_len; seq_id += threads) {
    target_pos = batch_offset + seq_id;
    pout_idx = temp_offset[pout] + batch_offset + seq_id;
    temp[pout_idx] =
        (seq_id > 0 && input[target_pos - 1] != padding_idx) ? 1 : 0;
  }
  __syncthreads();
  for (int stride = 1; stride < seq_len; stride *= 2) {
    pout = 1 - pout;
    pin = 1 - pout;
    for (int seq_id = start_seq_id; seq_id < seq_len; seq_id += threads) {
      pout_idx = temp_offset[pout] + batch_offset + seq_id;
      pin_idx = temp_offset[pin] + batch_offset + seq_id;
      if (seq_id >= stride)
        temp[pout_idx] = temp[pin_idx] + temp[pin_idx - stride];
      else
        temp[pout_idx] = temp[pin_idx];
    }
    __syncthreads();
  }

  for (int seq_id = start_seq_id; seq_id < seq_len; seq_id += threads) {
    target_pos = batch_offset + seq_id;
    pout_idx = temp_offset[pout] + batch_offset + seq_id;
    output[target_pos] = temp[pout_idx];
  }
}

__device__ int get_clip_mask(float value, float clip_max) {
  if (value >= clip_max) {
    return 2;
  } else if (value <= -clip_max) {
    return 4;
  } else {
    return 0;
  }
}

/**
@brief: lookup_scale_pos_dropout
forward of embedding layer in fairseq, including
lookup table, scale, add position embedding and dropout.

@thread
gridDim.x = batch_size
gridDim.y = blocks_per_seq
blockDim.x = tokens_per_block
blockDim.y = min(embedding_dim, MAX_THREADS)

@param
。
output: [batch_size, seq_len, embedding_dim]
input: [batch_size, seq_len]
tokens_position: [batch_size, seq_len]
embeddings: [vocab_size, embedding_dim]
pos_embeddings: [max_seq_len, embedding_dim]
dropout_mask: [batch_size, seq_len, embedding_dim]
batch_size: the size of the current batch
seq_len: the sequence length of the current batch
embedding_dim: dim of the embeddings
padding_idx: padding index of the sentences (default: 2)
step: only used to calculate correct position in inference (default: 0 in
training and valid)
*/
template <typename T>
__global__ void lookup_scale_pos_dropout(
    T *output, const int *input, const int *tokens_position,
    const T *embeddings, const T *pos_embeddings, const T *clip_max,
    uint8_t *dropout_mask, int seq_len, int embedding_dim, int padding_idx,
    float dropout_ratio, float emb_scale, int step, int seed);

template <>
__global__ void lookup_scale_pos_dropout<float>(
    float *output, const int *input, const int *tokens_position,
    const float *embeddings, const float *pos_embeddings, const float *clip_max,
    uint8_t *dropout_mask, int seq_len, int embedding_dim, int padding_idx,
    float dropout_ratio, float emb_scale, int step, int seed) {
  int batch_id = blockIdx.x;
  int seq_id = blockIdx.y * blockDim.x + threadIdx.x;
  if (seq_id >= seq_len) return;

  int target_pos = batch_id * seq_len + seq_id;
  int start = target_pos * embedding_dim + threadIdx.y;
  int end = (target_pos + 1) * embedding_dim;
  int tid = input[target_pos];

  int token_pos_id = tokens_position[target_pos];

  float4 *output4 = reinterpret_cast<float4 *>(output);
  const float4 *embeddings4 = reinterpret_cast<const float4 *>(embeddings);
  const float4 *pos_embeddings4 =
      reinterpret_cast<const float4 *>(pos_embeddings);
  uint32_t *dropout_mask4 = reinterpret_cast<uint32_t *>(dropout_mask);

  // no need to calculate dropout_mask
  if (tid == padding_idx) {
    float4 zero4;
    zero4.x = zero4.y = zero4.z = zero4.w = 0.f;
    for (uint i = start; i < end; i += blockDim.y) {
      output4[i] = zero4;
    }
    return;
  }

  const float dropout_scale = 1.f / (1.f - dropout_ratio);
  float clip_max_val;
  if (clip_max) {
    clip_max_val = clip_max[0];
  }
  curandStatePhilox4_32_10_t state;

  for (uint i = start; i < end; i += blockDim.y) {
    curand_init(seed, i, 0, &state);
    float4 rand4 = curand_uniform4(&state);
    uint8_t m[4];
    // dropout mask
    m[0] = (uint8_t)(rand4.x > dropout_ratio);
    m[1] = (uint8_t)(rand4.y > dropout_ratio);
    m[2] = (uint8_t)(rand4.z > dropout_ratio);
    m[3] = (uint8_t)(rand4.w > dropout_ratio);

    int offset = i - target_pos * embedding_dim;
    // step is non-zero only in inference
    float4 e4 = embeddings4[tid * embedding_dim + offset];
    float4 pe4 =
        pos_embeddings4[(token_pos_id + step) * embedding_dim + offset];
    float4 res4;

    float scale_mask[4];
    scale_mask[0] = dropout_scale * m[0];
    scale_mask[1] = dropout_scale * m[1];
    scale_mask[2] = dropout_scale * m[2];
    scale_mask[3] = dropout_scale * m[3];

    uint8_t clip_mask[4];
    if (clip_max) {
      e4.x = fake_quantize(e4.x, clip_max_val, clip_mask[0], 2);
      e4.y = fake_quantize(e4.y, clip_max_val, clip_mask[1], 2);
      e4.z = fake_quantize(e4.z, clip_max_val, clip_mask[2], 2);
      e4.w = fake_quantize(e4.w, clip_max_val, clip_mask[3], 2);
    }
    res4.x = (emb_scale * e4.x + pe4.x) * scale_mask[0];
    res4.y = (emb_scale * e4.y + pe4.y) * scale_mask[1];
    res4.z = (emb_scale * e4.z + pe4.z) * scale_mask[2];
    res4.w = (emb_scale * e4.w + pe4.w) * scale_mask[3];

    output4[i] = res4;
    uint32_t *m4 = reinterpret_cast<uint32_t *>(m);
    if (clip_max) {
      m4[0] = m4[0] | reinterpret_cast<uint32_t *>(clip_mask)[0];
    }
    dropout_mask4[i] = m4[0];
  }
}

template <>
__global__ void lookup_scale_pos_dropout<__half>(
    __half *output, const int *input, const int *tokens_position,
    const __half *embeddings, const __half *pos_embeddings,
    const __half *clip_max, uint8_t *dropout_mask, int seq_len,
    int embedding_dim, int padding_idx, float dropout_ratio, float emb_scale,
    int step, int seed) {
  int batch_id = blockIdx.x;
  int seq_id = blockIdx.y * blockDim.x + threadIdx.x;
  if (seq_id >= seq_len) return;

  int target_pos = batch_id * seq_len + seq_id;
  int start = target_pos * embedding_dim + threadIdx.y;
  int end = (target_pos + 1) * embedding_dim;
  int tid = input[target_pos];

  int token_pos_id = tokens_position[target_pos];

  float4 *output4 = reinterpret_cast<float4 *>(output);
  const float4 *embeddings4 = reinterpret_cast<const float4 *>(embeddings);
  const float4 *pos_embeddings4 =
      reinterpret_cast<const float4 *>(pos_embeddings);
  uint64_t *dropout_mask8 = reinterpret_cast<uint64_t *>(dropout_mask);

  // no need to calculate dropout_mask
  if (tid == padding_idx) {
    float4 zero4;
    zero4.x = zero4.y = zero4.z = zero4.w = 0.f;
    for (uint i = start; i < end; i += blockDim.y) {
      output4[i] = zero4;
    }
    return;
  }

  const float dropout_scale = 1.f / (1.f - dropout_ratio);
  float clip_max_val;
  if (clip_max) {
    clip_max_val = __half2float(clip_max[0]);
  }

  curandStatePhilox4_32_10_t state;

  for (uint i = start; i < end; i += blockDim.y) {
    curand_init(seed, i, 0, &state);
    float4 rand4 = curand_uniform4(&state);
    uint8_t m[8];
    m[0] = (uint8_t)(rand4.x > dropout_ratio);
    m[1] = (uint8_t)(rand4.y > dropout_ratio);
    m[2] = (uint8_t)(rand4.z > dropout_ratio);
    m[3] = (uint8_t)(rand4.w > dropout_ratio);
    rand4 = curand_uniform4(&state);
    m[4] = (uint8_t)(rand4.x > dropout_ratio);
    m[5] = (uint8_t)(rand4.y > dropout_ratio);
    m[6] = (uint8_t)(rand4.z > dropout_ratio);
    m[7] = (uint8_t)(rand4.w > dropout_ratio);

    int offset = i - target_pos * embedding_dim;
    float4 e4 = embeddings4[tid * embedding_dim + offset];
    // step is non-zero only in inference
    float4 pe4 =
        pos_embeddings4[(token_pos_id + step) * embedding_dim + offset];
    float4 res4;

    __half2 *e_h2 = reinterpret_cast<__half2 *>(&e4);
    __half2 *pe_h2 = reinterpret_cast<__half2 *>(&pe4);
    __half2 *res_h2 = reinterpret_cast<__half2 *>(&res4);
    __half2 scale_mask_h2[4];

#pragma unroll
    for (uint j = 0; j < 4; ++j) {
      scale_mask_h2[j] = __floats2half2_rn(dropout_scale * m[j << 1],
                                           dropout_scale * m[(j << 1) | 1]);
    }
    __half2 emb_scale_h2 = __floats2half2_rn(emb_scale, emb_scale);

    uint8_t clip_mask[8];
    for (uint j = 0; j < 4; ++j) {
      if (clip_max) {
        float2 f2 = __half22float2(e_h2[j]);
        // fake quant
        f2.x = fake_quantize(f2.x, clip_max_val, clip_mask[j << 1], 2);
        f2.y = fake_quantize(f2.y, clip_max_val, clip_mask[(j << 1) | 1], 2);
        res_h2[j] = __hmul2(__float22half2_rn(f2), emb_scale_h2);
      } else {
        res_h2[j] = __hmul2(e_h2[j], emb_scale_h2);
      }
      res_h2[j] = __hadd2(res_h2[j], pe_h2[j]);
      res_h2[j] = __hmul2(res_h2[j], scale_mask_h2[j]);
    }
    output4[i] = res4;
    uint64_t *m8 = reinterpret_cast<uint64_t *>(m);
    if (clip_max) {
      m8[0] = m8[0] | reinterpret_cast<uint64_t *>(clip_mask)[0];
    }
    dropout_mask8[i] = m8[0];
  }
}

template <>
void launch_lookup_scale_pos_dropout<float>(
    float *output, const int *input, const float *embeddings,
    const float *pos_embeddings, const float *clip_max, uint8_t *dropout_mask,
    int *tokens_position, int batch_size, int seq_len, int embedding_dim,
    int padding_idx, float dropout_ratio, int step, cudaStream_t &stream) {
  int p_threads = min(seq_len, MAX_THREADS);
  dim3 p_grid_dim(batch_size, 1);
  dim3 p_block_dim(p_threads, 1);
  // get the position index of the tokens alone,
  // because synchronization is required at the sequence level
  get_tokens_position<<<p_grid_dim, p_block_dim, 0, stream>>>(
      tokens_position, input, batch_size, seq_len, padding_idx);

  float emb_scale = sqrt(embedding_dim);
  embedding_dim >>= 2;

  int threads_per_token = min(embedding_dim, MAX_THREADS);
  // int tokens_per_block = MAX_THREADS / threads_per_token;
  int tokens_per_block = 1;
  int blocks_per_seq = (seq_len + tokens_per_block - 1) / tokens_per_block;
  dim3 grid_dim(batch_size, blocks_per_seq);
  dim3 block_dim(tokens_per_block, threads_per_token);
  int seed = std::chrono::duration_cast<std::chrono::microseconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
  lookup_scale_pos_dropout<float><<<grid_dim, block_dim, 0, stream>>>(
      output, input, tokens_position, embeddings, pos_embeddings, clip_max,
      dropout_mask, seq_len, embedding_dim, padding_idx, dropout_ratio,
      emb_scale, step, seed);
}

template <>
void launch_lookup_scale_pos_dropout<__half>(
    __half *output, const int *input, const __half *embeddings,
    const __half *pos_embeddings, const __half *clip_max, uint8_t *dropout_mask,
    int *tokens_position, int batch_size, int seq_len, int embedding_dim,
    int padding_idx, float dropout_ratio, int step, cudaStream_t &stream) {
  int p_threads = min(seq_len, MAX_THREADS);
  dim3 p_grid_dim(batch_size, 1);
  dim3 p_block_dim(p_threads, 1);
  // get the position index of the tokens alone,
  // because synchronization is required at the sequence level
  get_tokens_position<<<p_grid_dim, p_block_dim, 0, stream>>>(
      tokens_position, input, batch_size, seq_len, padding_idx);

  float emb_scale = sqrt(embedding_dim);
  embedding_dim >>= 3;

  int threads_per_token = min(embedding_dim, MAX_THREADS);
  // int tokens_per_block = MAX_THREADS / threads_per_token;
  int tokens_per_block = 1;
  int blocks_per_seq = (seq_len + tokens_per_block - 1) / tokens_per_block;
  dim3 grid_dim(batch_size, blocks_per_seq);
  dim3 block_dim(tokens_per_block, threads_per_token);
  int seed = std::chrono::duration_cast<std::chrono::microseconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
  lookup_scale_pos_dropout<__half><<<grid_dim, block_dim, 0, stream>>>(
      output, input, tokens_position, embeddings, pos_embeddings, clip_max,
      dropout_mask, seq_len, embedding_dim, padding_idx, dropout_ratio,
      emb_scale, step, seed);
}

/**
@brief: d_lookup_scale_pos_dropout
backward of embedding layer in fairseq.

@thread
gridDim.x = batch_size
gridDim.y = blocks_per_seq
blockDim.x = tokens_per_block
blockDim.y = min(embedding_dim, MAX_THREADS)

@param
input: [batch_size, seq_len]
grad_output: [batch_size, seq_len, embedding_dim]
dropout_mask: [batch_size, seq_len, embedding_dim]
batch_size: the size of the current batch
seq_len: the sequence length of the current batch
embedding_dim: dim of the embeddings
padding_idx: padding index of the sentences (default: 2)
*/
template <typename T>
__global__ void zero_grads(T *grad_embeddings, int total_nums) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_nums) return;
  float4 *grad_embeddings4 = reinterpret_cast<float4 *>(grad_embeddings);
  float4 zero4;
  zero4.x = zero4.y = zero4.z = zero4.w = 0.f;
  grad_embeddings4[idx] = zero4;
}

template <typename T>
__global__ void d_lookup_scale_pos_dropout(
    T *grad_embeddings, T *grad_clip_max, const T *grad_output,
    const int *input, const uint8_t *dropout_mask, int seq_len,
    int embedding_dim, int padding_idx, float dropout_ratio, float emb_scale);

template <>
__global__ void d_lookup_scale_pos_dropout<float>(
    float *grad_embeddings, float *grad_clip_max, const float *grad_output,
    const int *input, const uint8_t *dropout_mask, int seq_len,
    int embedding_dim, int padding_idx, float dropout_ratio, float emb_scale) {
  int batch_id = blockIdx.x;
  int seq_id = blockIdx.y * blockDim.x + threadIdx.x;
  if (seq_id >= seq_len) return;

  int target_pos = batch_id * seq_len + seq_id;
  int start = target_pos * embedding_dim + threadIdx.y;
  int end = (target_pos + 1) * embedding_dim;
  int tid = input[target_pos];

  if (tid == padding_idx) {
    return;
  }

  const float scale = 1.f / (1.f - dropout_ratio);
  const float4 *grad_output4 = reinterpret_cast<const float4 *>(grad_output);
  const uint32_t *dropout_mask4 =
      reinterpret_cast<const uint32_t *>(dropout_mask);
  // float block_g_clip_max = 0;
  float thread_cmax_grad = 0;
  float temp_cmax_grad = 0;
  for (uint i = start; i < end; i += blockDim.y) {
    float4 go4 = grad_output4[i];
    uint32_t m4 = dropout_mask4[i];
    uint8_t *m4_ptr = reinterpret_cast<uint8_t *>(&m4);
    float4 res4;
    res4.x = emb_scale * go4.x * (m4_ptr[0] & 1) * scale;
    res4.y = emb_scale * go4.y * (m4_ptr[1] & 1) * scale;
    res4.z = emb_scale * go4.z * (m4_ptr[2] & 1) * scale;
    res4.w = emb_scale * go4.w * (m4_ptr[3] & 1) * scale;
    int offset = i - target_pos * embedding_dim;
    int idx = (tid * (embedding_dim) + offset) << 2;
    clip_bwd(res4.x, temp_cmax_grad, res4.x, m4_ptr[0], 2);
    thread_cmax_grad += temp_cmax_grad;
    clip_bwd(res4.y, temp_cmax_grad, res4.y, m4_ptr[1], 2);
    thread_cmax_grad += temp_cmax_grad;
    clip_bwd(res4.z, temp_cmax_grad, res4.z, m4_ptr[2], 2);
    thread_cmax_grad += temp_cmax_grad;
    clip_bwd(res4.w, temp_cmax_grad, res4.w, m4_ptr[3], 2);
    thread_cmax_grad += temp_cmax_grad;
    atomicAdd(grad_embeddings + idx, res4.x);
    atomicAdd(grad_embeddings + idx + 1, res4.y);
    atomicAdd(grad_embeddings + idx + 2, res4.z);
    atomicAdd(grad_embeddings + idx + 3, res4.w);
  }

  if (grad_clip_max) {
    __shared__ float block_cmax_grad;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      block_cmax_grad = 0;
    }
    __syncthreads();
    if (thread_cmax_grad != 0) {
      atomicAdd(&block_cmax_grad, thread_cmax_grad);
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      if (block_cmax_grad != 0) {
        atomicAdd(&grad_clip_max[0], block_cmax_grad);
      }
    }
  }
}

template <>
__global__ void d_lookup_scale_pos_dropout<__half>(
    __half *grad_embeddings, __half *grad_clip_max, const __half *grad_output,
    const int *input, const uint8_t *dropout_mask, int seq_len,
    int embedding_dim, int padding_idx, float dropout_ratio, float emb_scale) {
  int batch_id = blockIdx.x;
  int seq_id = blockIdx.y * blockDim.x + threadIdx.x;
  if (seq_id >= seq_len) return;

  int target_pos = batch_id * seq_len + seq_id;
  int start = target_pos * embedding_dim + threadIdx.y;
  int end = (target_pos + 1) * embedding_dim;
  int tid = input[target_pos];

  if (tid == padding_idx) {
    return;
  }

  const float scale = 1.f / (1.f - dropout_ratio);
  const float4 *grad_output4 = reinterpret_cast<const float4 *>(grad_output);
  const uint64_t *dropout_mask4 =
      reinterpret_cast<const uint64_t *>(dropout_mask);
  __half2 *grad_embeddings_h2 = reinterpret_cast<__half2 *>(grad_embeddings);
  float block_g_clip_max = 0;
  float thread_cmax_grad = 0;
  float temp_cmax_grad = 0;
  for (uint i = start; i < end; i += blockDim.y) {
    float4 go4 = grad_output4[i];
    uint64_t m4 = dropout_mask4[i];
    uint8_t *m4_ptr = reinterpret_cast<uint8_t *>(&m4);
    float4 res4;
    __half2 *go_h2 = reinterpret_cast<__half2 *>(&go4);
    __half2 *res_h2 = reinterpret_cast<__half2 *>(&res4);
    __half2 scale_mask_h2[4];

#pragma unroll
    for (uint j = 0; j < 4; ++j) {
      scale_mask_h2[j] = __floats2half2_rn(scale * (m4_ptr[j << 1] & 1),
                                           scale * (m4_ptr[(j << 1) | 1] & 1));
    }
    __half2 emb_scale_h2 = __floats2half2_rn(emb_scale, emb_scale);

#pragma unroll
    for (uint j = 0; j < 4; ++j) {
      res_h2[j] = __hmul2(emb_scale_h2, go_h2[j]);
      res_h2[j] = __hmul2(scale_mask_h2[j], res_h2[j]);
    }

    int offset = i - target_pos * embedding_dim;
    int idx = (tid * (embedding_dim) + offset) << 2;
#pragma unroll
    for (uint j = 0; j < 4; ++j) {
      clip_bwd(res_h2[j].x, temp_cmax_grad, res_h2[j].x, m4_ptr[j << 1], 2);
      thread_cmax_grad += temp_cmax_grad;
      clip_bwd(res_h2[j].y, temp_cmax_grad, res_h2[j].y, m4_ptr[(j << 1) | 1],
               2);
      thread_cmax_grad += temp_cmax_grad;

      atomicAdd(grad_embeddings_h2 + idx + j, res_h2[j]);

      if (grad_clip_max) {
        block_g_clip_max +=
            __half2float(res_h2[j].x) * is_max_min_mask(m4_ptr[j << 1], 2) +
            __half2float(res_h2[j].y) *
                is_max_min_mask(m4_ptr[(j << 1) | 1], 2);
      }
    }
  }

  if (grad_clip_max) {
    __shared__ float block_cmax_grad;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      block_cmax_grad = 0;
    }
    __syncthreads();
    if (thread_cmax_grad != 0) {
      atomicAdd(&block_cmax_grad, thread_cmax_grad);
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      if (block_cmax_grad != 0) {
        atomicAdd(&grad_clip_max[0], block_cmax_grad);
      }
    }
  }
}

/**
@brief: d_lookup_scale_trainable_pos_dropout
backward of embedding layer in fairseq.

@thread
gridDim.x = batch_size
gridDim.y = blocks_per_seq
blockDim.x = tokens_per_block
blockDim.y = min(embedding_dim, MAX_THREADS)

@param
input: [batch_size, seq_len]
grad_output: [batch_size, seq_len, embedding_dim]
dropout_mask: [batch_size, seq_len, embedding_dim]
batch_size: the size of the current batch
seq_len: the sequence length of the current batch
embedding_dim: dim of the embeddings
padding_idx: padding index of the sentences (default: 2)
*/

template <typename T>
__global__ void d_lookup_scale_trainable_pos_dropout(
    T *grad_embeddings, const T *grad_output, T *grad_pos_embeddings,
    const int *input, const uint8_t *dropout_mask, const int *tokens_position,
    int seq_len, int embedding_dim, int padding_idx, float dropout_ratio,
    float emb_scale);

template <>
__global__ void d_lookup_scale_trainable_pos_dropout<float>(
    float *grad_embeddings, const float *grad_output,
    float *grad_pos_embeddings, const int *input, const uint8_t *dropout_mask,
    const int *tokens_position, int seq_len, int embedding_dim, int padding_idx,
    float dropout_ratio, float emb_scale) {
  int batch_id = blockIdx.x;
  int seq_id = blockIdx.y * blockDim.x + threadIdx.x;
  if (seq_id >= seq_len) return;

  int target_pos = batch_id * seq_len + seq_id;
  int start = target_pos * embedding_dim + threadIdx.y;
  int end = (target_pos + 1) * embedding_dim;
  int tid = input[target_pos];
  int token_pos_id = tokens_position[target_pos];

  if (tid == padding_idx) {
    return;
  }

  const float scale = 1.f / (1.f - dropout_ratio);
  const float4 *grad_output4 = reinterpret_cast<const float4 *>(grad_output);
  const uint32_t *dropout_mask4 =
      reinterpret_cast<const uint32_t *>(dropout_mask);

  for (uint i = start; i < end; i += blockDim.y) {
    float4 go4 = grad_output4[i];
    uint32_t m4 = dropout_mask4[i];
    uint8_t *m4_ptr = reinterpret_cast<uint8_t *>(&m4);

    float4 res4;
    res4.x = emb_scale * go4.x * m4_ptr[0] * scale;
    res4.y = emb_scale * go4.y * m4_ptr[1] * scale;
    res4.z = emb_scale * go4.z * m4_ptr[2] * scale;
    res4.w = emb_scale * go4.w * m4_ptr[3] * scale;
    int offset = i - target_pos * embedding_dim;
    int idx = (tid * (embedding_dim) + offset) << 2;
    atomicAdd(grad_embeddings + idx, res4.x);
    atomicAdd(grad_embeddings + idx + 1, res4.y);
    atomicAdd(grad_embeddings + idx + 2, res4.z);
    atomicAdd(grad_embeddings + idx + 3, res4.w);

    float4 p_res4;
    p_res4.x = go4.x * m4_ptr[0] * scale;
    p_res4.y = go4.y * m4_ptr[1] * scale;
    p_res4.z = go4.z * m4_ptr[2] * scale;
    p_res4.w = go4.w * m4_ptr[3] * scale;
    idx = (token_pos_id * (embedding_dim) + offset) << 2;
    atomicAdd(grad_pos_embeddings + idx, p_res4.x);
    atomicAdd(grad_pos_embeddings + idx + 1, p_res4.y);
    atomicAdd(grad_pos_embeddings + idx + 2, p_res4.z);
    atomicAdd(grad_pos_embeddings + idx + 3, p_res4.w);
  }
}

template <>
__global__ void d_lookup_scale_trainable_pos_dropout<__half>(
    __half *grad_embeddings, const __half *grad_output,
    __half *grad_pos_embeddings, const int *input, const uint8_t *dropout_mask,
    const int *tokens_position, int seq_len, int embedding_dim, int padding_idx,
    float dropout_ratio, float emb_scale) {
  int batch_id = blockIdx.x;
  int seq_id = blockIdx.y * blockDim.x + threadIdx.x;
  if (seq_id >= seq_len) return;

  int target_pos = batch_id * seq_len + seq_id;
  int start = target_pos * embedding_dim + threadIdx.y;
  int end = (target_pos + 1) * embedding_dim;
  int tid = input[target_pos];
  int token_pos_id = tokens_position[target_pos];

  if (tid == padding_idx) {
    return;
  }

  const float scale = 1.f / (1.f - dropout_ratio);
  const float4 *grad_output4 = reinterpret_cast<const float4 *>(grad_output);
  const uint64_t *dropout_mask4 =
      reinterpret_cast<const uint64_t *>(dropout_mask);
  __half2 *grad_embeddings_h2 = reinterpret_cast<__half2 *>(grad_embeddings);
  __half2 *grad_pos_embeddings_h2 =
      reinterpret_cast<__half2 *>(grad_pos_embeddings);

  for (uint i = start; i < end; i += blockDim.y) {
    float4 go4 = grad_output4[i];
    uint64_t m4 = dropout_mask4[i];
    uint8_t *m4_ptr = reinterpret_cast<uint8_t *>(&m4);
    float4 res4;
    __half2 *go_h2 = reinterpret_cast<__half2 *>(&go4);
    __half2 *res_h2 = reinterpret_cast<__half2 *>(&res4);
    __half2 scale_mask_h2[4];

#pragma unroll
    for (uint j = 0; j < 4; ++j) {
      scale_mask_h2[j] = __floats2half2_rn(scale * m4_ptr[j << 1],
                                           scale * m4_ptr[(j << 1) | 1]);
    }
    __half2 emb_scale_h2 = __floats2half2_rn(emb_scale, emb_scale);

#pragma unroll
    for (uint j = 0; j < 4; ++j) {
      res_h2[j] = __hmul2(emb_scale_h2, go_h2[j]);
      res_h2[j] = __hmul2(scale_mask_h2[j], res_h2[j]);
    }

    int offset = i - target_pos * embedding_dim;
    int idx = (tid * (embedding_dim) + offset) << 2;
#pragma unroll
    for (uint j = 0; j < 4; ++j) {
      atomicAdd(grad_embeddings_h2 + idx + j, res_h2[j]);
    }

#pragma unroll
    for (uint j = 0; j < 4; ++j) {
      res_h2[j] = __hmul2(scale_mask_h2[j], go_h2[j]);
    }

    idx = (token_pos_id * (embedding_dim) + offset) << 2;
#pragma unroll
    for (uint j = 0; j < 4; ++j) {
      atomicAdd(grad_pos_embeddings_h2 + idx + j, res_h2[j]);
    }
  }
}

template <>
void launch_d_lookup_scale_pos_dropout<float>(
    float *grad_embeddings, float *grad_clip_max, float *grad_pos_embeddings,
    const float *grad_output, const int *input, const uint8_t *dropout_mask,
    const int *tokens_position, int batch_size, int seq_len, int embedding_dim,
    int vocab_size, int max_seq_len, int padding_idx, float dropout_ratio,
    bool trainable_pos, cudaStream_t &stream) {
  float emb_scale = sqrt(embedding_dim);
  embedding_dim >>= 2;

  int total_nums = vocab_size * embedding_dim + 1;
  dim3 zg_grid_dim((total_nums + MAX_THREADS - 1) / MAX_THREADS);
  dim3 zg_block_dim(MAX_THREADS);

  zero_grads<float>
      <<<zg_grid_dim, zg_block_dim, 0, stream>>>(grad_embeddings, total_nums);

  int threads_per_token = min(embedding_dim, MAX_THREADS);
  // int tokens_per_block = MAX_THREADS / threads_per_token;
  int tokens_per_block = 1;
  int blocks_per_seq = (seq_len + tokens_per_block - 1) / tokens_per_block;
  dim3 grid_dim(batch_size, blocks_per_seq);
  dim3 block_dim(tokens_per_block, threads_per_token);

  if (trainable_pos) {
    total_nums = max_seq_len * embedding_dim;
    dim3 pos_grid_dim((total_nums + MAX_THREADS - 1) / MAX_THREADS);
    dim3 pos_block_dim(MAX_THREADS);

    zero_grads<float><<<pos_grid_dim, pos_block_dim, 0, stream>>>(
        grad_pos_embeddings, total_nums);

    d_lookup_scale_trainable_pos_dropout<float>
        <<<grid_dim, block_dim, 0, stream>>>(
            grad_embeddings, grad_output, grad_pos_embeddings, input,
            dropout_mask, tokens_position, seq_len, embedding_dim, padding_idx,
            dropout_ratio, emb_scale);
  } else {
    d_lookup_scale_pos_dropout<float><<<grid_dim, block_dim, 0, stream>>>(
        grad_embeddings, grad_clip_max, grad_output, input, dropout_mask,
        seq_len, embedding_dim, padding_idx, dropout_ratio, emb_scale);
  }
}

template <>
void launch_d_lookup_scale_pos_dropout<__half>(
    __half *grad_embeddings, __half *grad_clip_max, __half *grad_pos_embeddings,
    const __half *grad_output, const int *input, const uint8_t *dropout_mask,
    const int *tokens_position, int batch_size, int seq_len, int embedding_dim,
    int vocab_size, int max_seq_len, int padding_idx, float dropout_ratio,
    bool trainable_pos, cudaStream_t &stream) {
  float emb_scale = sqrt(embedding_dim);
  embedding_dim >>= 3;

  int total_nums = vocab_size * embedding_dim + 1;
  dim3 zg_grid_dim((total_nums + MAX_THREADS - 1) / MAX_THREADS);
  dim3 zg_block_dim(MAX_THREADS);

  zero_grads<__half>
      <<<zg_grid_dim, zg_block_dim, 0, stream>>>(grad_embeddings, total_nums);

  int threads_per_token = min(embedding_dim, MAX_THREADS);
  // int tokens_per_block = MAX_THREADS / threads_per_token;
  int tokens_per_block = 1;
  int blocks_per_seq = (seq_len + tokens_per_block - 1) / tokens_per_block;
  dim3 grid_dim(batch_size, blocks_per_seq);
  dim3 block_dim(tokens_per_block, threads_per_token);

  if (trainable_pos) {
    total_nums = max_seq_len * embedding_dim;
    dim3 pos_grid_dim((total_nums + MAX_THREADS - 1) / MAX_THREADS);
    dim3 pos_block_dim(MAX_THREADS);

    zero_grads<__half><<<pos_grid_dim, pos_block_dim, 0, stream>>>(
        grad_pos_embeddings, total_nums);

    d_lookup_scale_trainable_pos_dropout<__half>
        <<<grid_dim, block_dim, 0, stream>>>(
            grad_embeddings, grad_output, grad_pos_embeddings, input,
            dropout_mask, tokens_position, seq_len, embedding_dim, padding_idx,
            dropout_ratio, emb_scale);
  } else {
    d_lookup_scale_pos_dropout<__half><<<grid_dim, block_dim, 0, stream>>>(
        grad_embeddings, grad_clip_max, grad_output, input, dropout_mask,
        seq_len, embedding_dim, padding_idx, dropout_ratio, emb_scale);
  }
}
}  // namespace cuda
}  // namespace lightseq
