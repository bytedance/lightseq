#include <chrono>
#include <ctime>

#include "kernels.h"

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
    const T *embeddings, const T *pos_embeddings, uint8_t *dropout_mask,
    int seq_len, int embedding_dim, int padding_idx, float dropout_ratio,
    float emb_scale, int step, int seed);

template <>
__global__ void lookup_scale_pos_dropout<float>(
    float *output, const int *input, const int *tokens_position,
    const float *embeddings, const float *pos_embeddings, uint8_t *dropout_mask,
    int seq_len, int embedding_dim, int padding_idx, float dropout_ratio,
    float emb_scale, int step, int seed) {
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

  const float scale = 1.f / (1.f - dropout_ratio);
  curandStatePhilox4_32_10_t state;

  for (uint i = start; i < end; i += blockDim.y) {
    curand_init(seed, i, 0, &state);
    float4 rand4 = curand_uniform4(&state);
    uint8_t m[4];
    m[0] = (uint8_t)(rand4.x > dropout_ratio);
    m[1] = (uint8_t)(rand4.y > dropout_ratio);
    m[2] = (uint8_t)(rand4.z > dropout_ratio);
    m[3] = (uint8_t)(rand4.w > dropout_ratio);
    uint32_t *m4 = reinterpret_cast<uint32_t *>(m);
    dropout_mask4[i] = m4[0];

    int offset = i - target_pos * embedding_dim;
    float4 e4 = embeddings4[tid * embedding_dim + offset];
    // step is non-zero only in inference
    float4 pe4 =
        pos_embeddings4[(token_pos_id + step) * embedding_dim + offset];
    float4 res4;
    res4.x = (emb_scale * e4.x + pe4.x) * scale * m[0];
    res4.y = (emb_scale * e4.y + pe4.y) * scale * m[1];
    res4.z = (emb_scale * e4.z + pe4.z) * scale * m[2];
    res4.w = (emb_scale * e4.w + pe4.w) * scale * m[3];
    output4[i] = res4;
  }
}

template <>
__global__ void lookup_scale_pos_dropout<__half>(
    __half *output, const int *input, const int *tokens_position,
    const __half *embeddings, const __half *pos_embeddings,
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

  const float scale = 1.f / (1.f - dropout_ratio);
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
    uint64_t *m8 = reinterpret_cast<uint64_t *>(m);
    dropout_mask8[i] = m8[0];

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
      scale_mask_h2[j] =
          __floats2half2_rn(scale * m[j << 1], scale * m[(j << 1) | 1]);
    }
    __half2 emb_scale_h2 = __floats2half2_rn(emb_scale, emb_scale);

#pragma unroll
    for (uint j = 0; j < 4; ++j) {
      res_h2[j] = __hmul2(e_h2[j], emb_scale_h2);
      res_h2[j] = __hadd2(res_h2[j], pe_h2[j]);
      res_h2[j] = __hmul2(res_h2[j], scale_mask_h2[j]);
    }
    output4[i] = res4;
  }
}

template <>
void launch_lookup_scale_pos_dropout<float>(
    float *output, const int *input, const float *embeddings,
    const float *pos_embeddings, uint8_t *dropout_mask, int batch_size,
    int seq_len, int embedding_dim, int padding_idx, float dropout_ratio,
    int step, cudaStream_t &stream) {
  int *tokens_position;
  cudaMalloc(&tokens_position, batch_size * seq_len * 2 * sizeof(int));
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
  int tokens_per_block = MAX_THREADS / threads_per_token;
  int blocks_per_seq = (seq_len + tokens_per_block - 1) / tokens_per_block;
  dim3 grid_dim(batch_size, blocks_per_seq);
  dim3 block_dim(tokens_per_block, threads_per_token);
  int seed = std::chrono::duration_cast<std::chrono::microseconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
  lookup_scale_pos_dropout<float><<<grid_dim, block_dim, 0, stream>>>(
      output, input, tokens_position, embeddings, pos_embeddings, dropout_mask,
      seq_len, embedding_dim, padding_idx, dropout_ratio, emb_scale, step,
      seed);

  cudaFree(tokens_position);
}

template <>
void launch_lookup_scale_pos_dropout<__half>(
    __half *output, const int *input, const __half *embeddings,
    const __half *pos_embeddings, uint8_t *dropout_mask, int batch_size,
    int seq_len, int embedding_dim, int padding_idx, float dropout_ratio,
    int step, cudaStream_t &stream) {
  int *tokens_position;
  cudaMalloc(&tokens_position, batch_size * seq_len * 2 * sizeof(int));
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
  int tokens_per_block = MAX_THREADS / threads_per_token;
  int blocks_per_seq = (seq_len + tokens_per_block - 1) / tokens_per_block;
  dim3 grid_dim(batch_size, blocks_per_seq);
  dim3 block_dim(tokens_per_block, threads_per_token);
  int seed = std::chrono::duration_cast<std::chrono::microseconds>(
                 std::chrono::system_clock::now().time_since_epoch())
                 .count();
  lookup_scale_pos_dropout<__half><<<grid_dim, block_dim, 0, stream>>>(
      output, input, tokens_position, embeddings, pos_embeddings, dropout_mask,
      seq_len, embedding_dim, padding_idx, dropout_ratio, emb_scale, step,
      seed);

  cudaFree(tokens_position);
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
    T *grad_embeddings, const T *grad_output, const int *input,
    const uint8_t *dropout_mask, int seq_len, int embedding_dim,
    int padding_idx, float dropout_ratio, float emb_scale);

template <>
__global__ void d_lookup_scale_pos_dropout<float>(
    float *grad_embeddings, const float *grad_output, const int *input,
    const uint8_t *dropout_mask, int seq_len, int embedding_dim,
    int padding_idx, float dropout_ratio, float emb_scale) {
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
  }
}

template <>
__global__ void d_lookup_scale_pos_dropout<__half>(
    __half *grad_embeddings, const __half *grad_output, const int *input,
    const uint8_t *dropout_mask, int seq_len, int embedding_dim,
    int padding_idx, float dropout_ratio, float emb_scale) {
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
  }
}

template <>
void launch_d_lookup_scale_pos_dropout<float>(
    float *grad_embeddings, const float *grad_output, const int *input,
    const uint8_t *dropout_mask, int batch_size, int seq_len, int embedding_dim,
    int vocab_size, int padding_idx, float dropout_ratio,
    cudaStream_t &stream) {
  float emb_scale = sqrt(embedding_dim);
  embedding_dim >>= 2;

  int total_nums = vocab_size * embedding_dim;
  dim3 zg_grid_dim((total_nums + MAX_THREADS - 1) / MAX_THREADS);
  dim3 zg_block_dim(MAX_THREADS);

  zero_grads<float>
      <<<zg_grid_dim, zg_block_dim, 0, stream>>>(grad_embeddings, total_nums);

  int threads_per_token = min(embedding_dim, MAX_THREADS);
  int tokens_per_block = MAX_THREADS / threads_per_token;
  int blocks_per_seq = (seq_len + tokens_per_block - 1) / tokens_per_block;
  dim3 grid_dim(batch_size, blocks_per_seq);
  dim3 block_dim(tokens_per_block, threads_per_token);

  d_lookup_scale_pos_dropout<float><<<grid_dim, block_dim, 0, stream>>>(
      grad_embeddings, grad_output, input, dropout_mask, seq_len, embedding_dim,
      padding_idx, dropout_ratio, emb_scale);
}

template <>
void launch_d_lookup_scale_pos_dropout<__half>(
    __half *grad_embeddings, const __half *grad_output, const int *input,
    const uint8_t *dropout_mask, int batch_size, int seq_len, int embedding_dim,
    int vocab_size, int padding_idx, float dropout_ratio,
    cudaStream_t &stream) {
  float emb_scale = sqrt(embedding_dim);
  embedding_dim >>= 3;

  int total_nums = vocab_size * embedding_dim;
  dim3 zg_grid_dim((total_nums + MAX_THREADS - 1) / MAX_THREADS);
  dim3 zg_block_dim(MAX_THREADS);

  zero_grads<__half>
      <<<zg_grid_dim, zg_block_dim, 0, stream>>>(grad_embeddings, total_nums);

  int threads_per_token = min(embedding_dim, MAX_THREADS);
  int tokens_per_block = MAX_THREADS / threads_per_token;
  int blocks_per_seq = (seq_len + tokens_per_block - 1) / tokens_per_block;
  dim3 grid_dim(batch_size, blocks_per_seq);
  dim3 block_dim(tokens_per_block, threads_per_token);

  d_lookup_scale_pos_dropout<__half><<<grid_dim, block_dim, 0, stream>>>(
      grad_embeddings, grad_output, input, dropout_mask, seq_len, embedding_dim,
      padding_idx, dropout_ratio, emb_scale);
}
