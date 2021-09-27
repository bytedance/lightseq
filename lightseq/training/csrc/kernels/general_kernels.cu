#include "kernels.h"
#include <iostream>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

/**
@brief: fuse_transpose_bias
Calculate the sum of elements in each column of the matrix.

@thread
gridDim.x = ceil(cols / WARP_SIZE)
blockDim.x = WARP_SIZE
blockDim.y = WARP_SIZE

@param
inp: [rows, cols]
out: [cols]
rows: the number of rows in the matrix
cols: the number of cols in the matrix
*/
template <typename T>
__global__ void column_sum_reduce(const T *__restrict__ inp,
                                  T *__restrict__ out, int rows, int cols) {
  __shared__ float tile[WARP_SIZE][WARP_SIZE];

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

  int idx = flat_2dim(blockIdx.x, threadIdx.x, WARP_SIZE);
  int y_stride = cols * WARP_SIZE;
  float localSum = 0;

  // Loop across matrix row
  // TODO: optimize to log complexity
  if (idx < cols) {
    int offset = flat_2dim(threadIdx.y, idx, cols);
    for (int r = threadIdx.y; r < rows; r += WARP_SIZE) {
      localSum += (float)inp[offset];
      offset += y_stride;
    }
  }

  // The sum of a row in tile is equal to the sum of a col in original matrix
  tile[threadIdx.x][threadIdx.y] = localSum;

  __syncthreads();

  // Sum the shared buffer.
  // The change of threadIdx.x is continuous
  float sum = tile[threadIdx.y][threadIdx.x];

  __syncthreads();

  // Calculate the sum of a row in tile
  for (int i = 1; i < WARP_SIZE; i <<= 1) sum += g.shfl_down(sum, i);

  if (threadIdx.x == 0) {
    int pos = flat_2dim(blockIdx.x, threadIdx.y, WARP_SIZE);
    if (pos < cols) out[pos] = sum;
  }
}

// [r, c] -> [c]
template <>
void launch_fuse_transpose_bias_kernel<float>(const float *inp, float *out,
                                              int rows, int cols,
                                              cudaStream_t stream) {
  dim3 grid_dim((cols - 1) / WARP_SIZE + 1);
  dim3 block_dim(WARP_SIZE, WARP_SIZE);

  column_sum_reduce<float>
      <<<grid_dim, block_dim, 0, stream>>>(inp, out, rows, cols);
}

template <>
void launch_fuse_transpose_bias_kernel<__half>(const __half *inp, __half *out,
                                               int rows, int cols,
                                               cudaStream_t stream) {
  dim3 grid_dim((cols - 1) / WARP_SIZE + 1);
  dim3 block_dim(WARP_SIZE, WARP_SIZE);

  column_sum_reduce<__half>
      <<<grid_dim, block_dim, 0, stream>>>(inp, out, rows, cols);
}

/**
@brief: fused_add2
Add two matrix inp1 and inp2 to out.

@thread
gridDim.x = batch_size * seq_len
blockDim.x = min(hidden_dim, MAX_THREADS)

@param
inp1: [batch_size, seq_len, hidden_dim]
inp2: [batch_size, seq_len, hidden_dim]
out: [batch_size, seq_len, hidden_dim]
batch_size: the size of the current batch
seq_len: the sequence length of the current batch
hidden_dim: dim of the hidden tensor
*/
template <typename T>
__global__ void fused_add2_kernel(T *out, const T *inp1, const T *inp2,
                                  int hidden_dim);

template <>
__global__ void fused_add2_kernel<float>(float *out, const float *inp1,
                                         const float *inp2, int hidden_dim) {
  int row_id = blockIdx.x;
  int offset = flat_2dim(row_id, 0, hidden_dim);

  const float4 *inp1_4 = reinterpret_cast<const float4 *>(inp1);
  const float4 *inp2_4 = reinterpret_cast<const float4 *>(inp2);
  float4 *out_4 = reinterpret_cast<float4 *>(out);
  float4 vinp1;
  float4 vinp2;
  float4 val;

  for (std::size_t i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
    vinp1 = inp1_4[offset + i];
    vinp2 = inp2_4[offset + i];
    val.x = vinp1.x + vinp2.x;
    val.y = vinp1.y + vinp2.y;
    val.z = vinp1.z + vinp2.z;
    val.w = vinp1.w + vinp2.w;
    out_4[offset + i] = val;
  }
}

template <>
__global__ void fused_add2_kernel<__half>(__half *out, const __half *inp1,
                                          const __half *inp2, int hidden_dim) {
  int row_id = blockIdx.x;
  int offset = flat_2dim(row_id, 0, hidden_dim);

  const float4 *inp1_4 = reinterpret_cast<const float4 *>(inp1);
  const float4 *inp2_4 = reinterpret_cast<const float4 *>(inp2);
  float4 *out_4 = reinterpret_cast<float4 *>(out);
  float4 vinp1;
  float4 vinp2;
  float4 val;
  __half2 *h2_inp1 = reinterpret_cast<__half2 *>(&vinp1);
  __half2 *h2_inp2 = reinterpret_cast<__half2 *>(&vinp2);
  __half2 *h2_val = reinterpret_cast<__half2 *>(&val);

  for (std::size_t i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
    vinp1 = inp1_4[offset + i];
    vinp2 = inp2_4[offset + i];
    h2_val[0] = __hadd2(h2_inp1[0], h2_inp2[0]);
    h2_val[1] = __hadd2(h2_inp1[1], h2_inp2[1]);
    h2_val[2] = __hadd2(h2_inp1[2], h2_inp2[2]);
    h2_val[3] = __hadd2(h2_inp1[3], h2_inp2[3]);
    out_4[offset + i] = val;
  }
}

//[b, s, h] -> [b, s, h]
template <>
void launch_fused_add2<float>(float *out, const float *inp1, const float *inp2,
                              int batch_size, int seq_len, int hidden_dim,
                              cudaStream_t &stream) {
  hidden_dim >>= 2;

  dim3 grid_dim(batch_size * seq_len);
  dim3 block_dim(min(hidden_dim, MAX_THREADS));

  fused_add2_kernel<<<grid_dim, block_dim, 0, stream>>>(out, inp1, inp2,
                                                        hidden_dim);
}

template <>
void launch_fused_add2<__half>(__half *out, const __half *inp1,
                               const __half *inp2, int batch_size, int seq_len,
                               int hidden_dim, cudaStream_t &stream) {
  hidden_dim >>= 3;

  dim3 grid_dim(batch_size * seq_len);
  dim3 block_dim(min(hidden_dim, MAX_THREADS));

  fused_add2_kernel<<<grid_dim, block_dim, 0, stream>>>(out, inp1, inp2,
                                                        hidden_dim);
}

template <typename T>
__global__ void kernel_concat3_dim1(const T *inp1, const T *inp2, T *output,
                                    int sz0, int sz2, int sz1_1, int sz1_2) {
  int nele = sz0 * sz2 * (sz1_1 + sz1_2);
  int idx = flat_2dim(blockIdx.x, threadIdx.x, blockDim.x);
  if (idx >= nele) {
    return;
  }
  float4 *dst_ptr = (float4 *)output + idx;
  int idx2 = idx % sz2;
  idx = idx / sz2;
  int idx1 = idx % (sz1_1 + sz1_2);
  int idx0 = idx / (sz1_1 + sz1_2);
  float4 *src_ptr = nullptr;
  int sz1 = 0;
  if (idx1 < sz1_1) {
    sz1 = sz1_1;
    src_ptr = (float4 *)inp1;
  } else {
    idx1 -= sz1_1;
    sz1 = sz1_2;
    src_ptr = (float4 *)inp2;
  }
  src_ptr += flat_3dim(idx0, idx1, idx2, sz1, sz2);
  dst_ptr[0] = src_ptr[0];
}

template <>
void launch_concat3_dim1<float>(const float *inp1, const float *inp2,
                                float *output, int sz0, int sz2, int sz1_1,
                                int sz1_2, cudaStream_t stream) {
  sz2 >>= 2;
  int nele = sz0 * sz2 * (sz1_1 + sz1_2);
  int nblock = (nele + MAX_THREADS - 1) / MAX_THREADS;
  kernel_concat3_dim1<<<nblock, MAX_THREADS, 0, stream>>>(
      inp1, inp2, output, sz0, sz2, sz1_1, sz1_2);
}

template <>
void launch_concat3_dim1<__half>(const __half *inp1, const __half *inp2,
                                 __half *output, int sz0, int sz2, int sz1_1,
                                 int sz1_2, cudaStream_t stream) {
  sz2 >>= 3;
  int nele = sz0 * sz2 * (sz1_1 + sz1_2);
  int nblock = (nele + MAX_THREADS - 1) / MAX_THREADS;
  kernel_concat3_dim1<<<nblock, MAX_THREADS, 0, stream>>>(
      inp1, inp2, output, sz0, sz2, sz1_1, sz1_2);
}

/**
@brief: ker_split_multilg_request
request = numpy.concatenate((src_lang_id, trg_lang_id, src_token_id), axis=1)

@thread
gridDim.x = (nele + MAX_THREADS - 1) / MAX_THREADS
blockDim.x = MAX_THREADS

@param
inp1: [batch_size, seq_len, hidden_dim]
inp2: [batch_size, seq_len, hidden_dim]
out: [batch_size, seq_len, hidden_dim]
batch_size: the size of the current batch
seq_len: the sequence length of the current batch
hidden_dim: dim of the hidden tensor
*/
__global__ void ker_split_multilg_request(const int *req, int *src_lang_id,
                                          int *trg_lang_id, int *src_token_id,
                                          int batch_size, int req_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < batch_size * req_len) {
    int value = req[idx];
    int seq_id = idx / req_len;
    int token_id = idx % req_len;

    if (token_id == 0) {
      src_lang_id[seq_id] = value;
    } else if (token_id == 1) {
      trg_lang_id[seq_id] = value;
    } else {
      int new_idx = flat_2dim(seq_id, token_id - 2, req_len - 2);
      src_token_id[new_idx] = value;
    }
  }
}

void launch_split_multilg_request(const int *req, int *src_lang_id,
                                  int *trg_lang_id, int *src_token_id,
                                  int batch_size, int req_len,
                                  cudaStream_t &stream) {
  if (req_len < 3) {
    throw std::runtime_error("req_len should be greater than 2");
  }
  int nele = batch_size * req_len;
  int nblock = (nele + MAX_THREADS - 1) / MAX_THREADS;
  ker_split_multilg_request<<<nblock, MAX_THREADS, 0, stream>>>(
      req, src_lang_id, trg_lang_id, src_token_id, batch_size, req_len);
}

/**
@brief: ker_enc_emb
for encoder, look up token embedding, add position embedding

@thread
gridDim.x = (nele + MAX_THREADS - 1) / MAX_THREADS
blockDim.x = MAX_THREADS;

@param
token_emb: [vocab_size, hidden_dim]
pos_emb: [max_step, hidden_dim]
token_id: input token id, [batch_size, seq_len]
output: result, [batch_size, seq_len, hidden_dim]
pad_mask: record the padding token, [batch_size, seq_len]
pad_id, the padding token id
*/
template <typename T>
__global__ void ker_enc_emb(const T *token_emb, const T *pos_emb,
                            const int *tokens, T *output, int *pad_mask,
                            int pad_id, int batch_size, int seq_len,
                            int hidden_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size * seq_len * hidden_dim) {
    return;
  }
  int batch_idx, seq_idx, dim_idx;
  decompose_3dim(idx, seq_len, hidden_dim, &batch_idx, &seq_idx, &dim_idx);
  int tokens_idx = batch_idx * seq_len + seq_idx;
  int token = tokens[tokens_idx];
  float4 value;

  if (token == pad_id) {
    if (dim_idx == 0) {
      pad_mask[tokens_idx] = 1;
    }
    value.x = 0.f;
    value.y = 0.f;
    value.z = 0.f;
    value.w = 0.f;
  } else {
    if (dim_idx == 0) {
      pad_mask[tokens_idx] = 0;
    }
    value = ((float4 *)token_emb)[token * hidden_dim + dim_idx];
    float4 pemb = ((float4 *)pos_emb)[seq_idx * hidden_dim + dim_idx];
    value.x += pemb.x;
    value.y += pemb.y;
    value.z += pemb.z;
    value.w += pemb.w;
  }
  ((float4 *)output)[idx] = value;
}

template <>
__global__ void ker_enc_emb<__half>(const __half *token_emb,
                                    const __half *pos_emb, const int *tokens,
                                    __half *output, int *pad_mask, int pad_id,
                                    int batch_size, int seq_len,
                                    int hidden_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size * seq_len * hidden_dim) {
    return;
  }
  int batch_idx, seq_idx, dim_idx;
  decompose_3dim(idx, seq_len, hidden_dim, &batch_idx, &seq_idx, &dim_idx);
  int tokens_idx = batch_idx * seq_len + seq_idx;
  int token = tokens[tokens_idx];
  float4 value;

  if (token == pad_id) {
    if (dim_idx == 0) {
      pad_mask[tokens_idx] = 1;
    }
    value.x = 0.f;
    value.y = 0.f;
    value.z = 0.f;
    value.w = 0.f;
  } else {
    if (dim_idx == 0) {
      pad_mask[tokens_idx] = 0;
    }
    value = ((float4 *)token_emb)[token * hidden_dim + dim_idx];
    float4 pemb = ((float4 *)pos_emb)[seq_idx * hidden_dim + dim_idx];
    __half2 *value_h2 = (__half2 *)(&value);
    __half2 *pemb_h2 = (__half2 *)(&pemb);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 value_f2 = __half22float2(value_h2[i]);
      float2 pemb_f2 = __half22float2(pemb_h2[i]);
      value_f2.x += pemb_f2.x;
      value_f2.y += pemb_f2.y;
      value_h2[i] = __float22half2_rn(value_f2);
    }
  }
  ((float4 *)output)[idx] = value;
}

template <typename T>
void launch_enc_emb(const T *token_emb, const T *pos_emb, const int *tokens,
                    T *output, int *pad_mask, int pad_id, int batch_size,
                    int seq_len, int hidden_dim, cudaStream_t stream) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  hidden_dim >>= 2;
  int nele = batch_size * seq_len * hidden_dim;
  int nblock = (nele + MAX_THREADS - 1) / MAX_THREADS;

  ker_enc_emb<T><<<nblock, MAX_THREADS, 0, stream>>>(
      token_emb, pos_emb, tokens, output, pad_mask, pad_id, batch_size, seq_len,
      hidden_dim);
}

template <>
void launch_enc_emb<__half>(const __half *token_emb, const __half *pos_emb,
                            const int *tokens, __half *output, int *pad_mask,
                            int pad_id, int batch_size, int seq_len,
                            int hidden_dim, cudaStream_t stream) {
  if (hidden_dim % 8 != 0) {
    throw std::runtime_error("violate hidden_dim % 8 = 0");
  }
  hidden_dim >>= 3;
  int nele = batch_size * seq_len * hidden_dim;
  int nblock = (nele + MAX_THREADS - 1) / MAX_THREADS;

  ker_enc_emb<__half><<<nblock, MAX_THREADS, 0, stream>>>(
      token_emb, pos_emb, tokens, output, pad_mask, pad_id, batch_size, seq_len,
      hidden_dim);
}

template void launch_enc_emb<float>(const float *token_emb,
                                    const float *pos_emb, const int *tokens,
                                    float *output, int *pad_mask, int pad_id,
                                    int batch_size, int seq_len, int hidden_dim,
                                    cudaStream_t stream);

template void launch_enc_emb<__half>(const __half *token_emb,
                                     const __half *pos_emb, const int *tokens,
                                     __half *output, int *pad_mask, int pad_id,
                                     int batch_size, int seq_len,
                                     int hidden_dim, cudaStream_t stream);

/**
@brief: ker_enc_emb_multilg_token
for encoder, look up token embedding, add position embedding

@thread
gridDim.x = (nele + MAX_THREADS - 1) / MAX_THREADS
blockDim.x = MAX_THREADS;

@param
token_emb: [vocab_size, hidden_dim]
pos_emb: [max_step, hidden_dim]
token_id: input token id, [batch_size, seq_len]
output: result, [batch_size, seq_len, hidden_dim]
pad_mask: record the padding token, [batch_size, seq_len]
pad_id, the padding token id
*/
template <typename T>
__global__ void ker_enc_emb_multilg_token(const T *token_emb, const T *pos_emb,
                                          const int *tokens, const T *lang_emb,
                                          const int *lang_id, T *output,
                                          int *pad_mask, int pad_id,
                                          int batch_size, int seq_len,
                                          int hidden_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size * seq_len * hidden_dim) {
    return;
  }
  int batch_idx, seq_idx, dim_idx;
  decompose_3dim(idx, seq_len, hidden_dim, &batch_idx, &seq_idx, &dim_idx);
  int tokens_idx = batch_idx * seq_len + seq_idx;
  int token = tokens[tokens_idx];
  float4 value;

  if (token == pad_id) {
    if (dim_idx == 0) {
      pad_mask[tokens_idx] = 1;
    }
    value.x = 0.f;
    value.y = 0.f;
    value.z = 0.f;
    value.w = 0.f;
  } else {
    if (dim_idx == 0) {
      pad_mask[tokens_idx] = 0;
    }
    value = ((float4 *)token_emb)[token * hidden_dim + dim_idx];

    // add pos emb
    float4 pemb = ((float4 *)pos_emb)[seq_idx * hidden_dim + dim_idx];
    value.x += pemb.x;
    value.y += pemb.y;
    value.z += pemb.z;
    value.w += pemb.w;
    // add lang emb
    pemb = ((float4 *)lang_emb)[lang_id[batch_idx] * hidden_dim + dim_idx];
    value.x += pemb.x;
    value.y += pemb.y;
    value.z += pemb.z;
    value.w += pemb.w;
  }
  ((float4 *)output)[idx] = value;
}

template <>
__global__ void ker_enc_emb_multilg_token<__half>(
    const __half *token_emb, const __half *pos_emb, const int *tokens,
    const __half *lang_emb, const int *lang_id, __half *output, int *pad_mask,
    int pad_id, int batch_size, int seq_len, int hidden_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size * seq_len * hidden_dim) {
    return;
  }
  int batch_idx, seq_idx, dim_idx;
  decompose_3dim(idx, seq_len, hidden_dim, &batch_idx, &seq_idx, &dim_idx);
  int tokens_idx = batch_idx * seq_len + seq_idx;
  int token = tokens[tokens_idx];
  float4 value;

  if (token == pad_id) {
    if (dim_idx == 0) {
      pad_mask[tokens_idx] = 1;
    }
    value.x = 0.f;
    value.y = 0.f;
    value.z = 0.f;
    value.w = 0.f;
  } else {
    if (dim_idx == 0) {
      pad_mask[tokens_idx] = 0;
    }
    value = ((float4 *)token_emb)[token * hidden_dim + dim_idx];
    __half2 *value_h2 = (__half2 *)(&value);

    float4 pemb = ((float4 *)pos_emb)[seq_idx * hidden_dim + dim_idx];
    __half2 *pemb_h2 = (__half2 *)(&pemb);
    float4 lemb =
        ((float4 *)lang_emb)[lang_id[batch_idx] * hidden_dim + dim_idx];
    __half2 *lemb_h2 = (__half2 *)(&lemb);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 value_f2 = __half22float2(value_h2[i]);
      float2 pemb_f2 = __half22float2(pemb_h2[i]);
      float2 lemb_f2 = __half22float2(lemb_h2[i]);
      value_f2.x += pemb_f2.x + lemb_f2.x;
      value_f2.y += pemb_f2.y + lemb_f2.y;
      value_h2[i] = __float22half2_rn(value_f2);
    }
  }
  ((float4 *)output)[idx] = value;
}

template <typename T>
void launch_enc_emb_multilg_token(const T *token_emb, const T *pos_emb,
                                  const int *tokens, const T *lang_emb,
                                  const int *lang_id, T *output, int *pad_mask,
                                  int pad_id, int batch_size, int seq_len,
                                  int hidden_dim, cudaStream_t stream) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  hidden_dim >>= 2;
  int nele = batch_size * seq_len * hidden_dim;
  int nblock = (nele + MAX_THREADS - 1) / MAX_THREADS;

  ker_enc_emb_multilg_token<T><<<nblock, MAX_THREADS, 0, stream>>>(
      token_emb, pos_emb, tokens, lang_emb, lang_id, output, pad_mask, pad_id,
      batch_size, seq_len, hidden_dim);
}

template <>
void launch_enc_emb_multilg_token<__half>(
    const __half *token_emb, const __half *pos_emb, const int *tokens,
    const __half *lang_emb, const int *lang_id, __half *output, int *pad_mask,
    int pad_id, int batch_size, int seq_len, int hidden_dim,
    cudaStream_t stream) {
  if (hidden_dim % 8 != 0) {
    throw std::runtime_error("violate hidden_dim % 8 = 0");
  }
  hidden_dim >>= 3;
  int nele = batch_size * seq_len * hidden_dim;
  int nblock = (nele + MAX_THREADS - 1) / MAX_THREADS;

  ker_enc_emb_multilg_token<__half><<<nblock, MAX_THREADS, 0, stream>>>(
      token_emb, pos_emb, tokens, lang_emb, lang_id, output, pad_mask, pad_id,
      batch_size, seq_len, hidden_dim);
}

template void launch_enc_emb_multilg_token<float>(
    const float *token_emb, const float *pos_emb, const int *tokens,
    const float *lang_emb, const int *lang_id, float *output, int *pad_mask,
    int pad_id, int batch_size, int seq_len, int hidden_dim,
    cudaStream_t stream);

template void launch_enc_emb_multilg_token<__half>(
    const __half *token_emb, const __half *pos_emb, const int *tokens,
    const __half *lang_emb, const int *lang_id, __half *output, int *pad_mask,
    int pad_id, int batch_size, int seq_len, int hidden_dim,
    cudaStream_t stream);

/**
@brief: ker_enc_emb_multilg_sentence
for encoder, look up token embedding, add position embedding

@thread
gridDim.x = (nele + MAX_THREADS - 1) / MAX_THREADS
blockDim.x = MAX_THREADS;

@param
token_emb: [vocab_size, hidden_dim]
pos_emb: [max_step, hidden_dim]
token_id: input token id, [batch_size, seq_len]
output: result, [batch_size, seq_len, hidden_dim]
pad_mask: record the padding token, [batch_size, seq_len]
pad_id, the padding token id
*/
template <typename T>
__global__ void ker_enc_emb_multilg_sentence(
    const T *token_emb, const T *pos_emb, const int *tokens, const T *lang_emb,
    const int *lang_id, T *output, int *pad_mask, int pad_id, int batch_size,
    int seq_len, int hidden_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size * seq_len * hidden_dim) {
    return;
  }
  int batch_idx, seq_idx, dim_idx;
  decompose_3dim(idx, seq_len, hidden_dim, &batch_idx, &seq_idx, &dim_idx);

  bool is_pad;
  int token_emb_idx;
  if (seq_idx == 0) {
    is_pad = false;
    token_emb = lang_emb;
    token_emb_idx = lang_id[batch_idx];
  } else {
    token_emb_idx = tokens[batch_idx * (seq_len - 1) + seq_idx - 1];
    is_pad = (token_emb_idx == pad_id);
  }

  float4 value;
  int tokens_idx = batch_idx * seq_len + seq_idx;
  if (is_pad) {
    if (dim_idx == 0) {
      pad_mask[tokens_idx] = 1;
    }
    value.x = 0.f;
    value.y = 0.f;
    value.z = 0.f;
    value.w = 0.f;
  } else {
    if (dim_idx == 0) {
      pad_mask[tokens_idx] = 0;
    }
    value = ((float4 *)token_emb)[token_emb_idx * hidden_dim + dim_idx];
    float4 pemb = ((float4 *)pos_emb)[seq_idx * hidden_dim + dim_idx];
    value.x += pemb.x;
    value.y += pemb.y;
    value.z += pemb.z;
    value.w += pemb.w;
  }
  ((float4 *)output)[idx] = value;
}

template <>
__global__ void ker_enc_emb_multilg_sentence<__half>(
    const __half *token_emb, const __half *pos_emb, const int *tokens,
    const __half *lang_emb, const int *lang_id, __half *output, int *pad_mask,
    int pad_id, int batch_size, int seq_len, int hidden_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size * seq_len * hidden_dim) {
    return;
  }
  int batch_idx, seq_idx, dim_idx;
  decompose_3dim(idx, seq_len, hidden_dim, &batch_idx, &seq_idx, &dim_idx);

  bool is_pad;
  int token_emb_idx;
  if (seq_idx == 0) {
    is_pad = false;
    token_emb = lang_emb;
    token_emb_idx = lang_id[batch_idx];
  } else {
    token_emb_idx = tokens[batch_idx * (seq_len - 1) + seq_idx - 1];
    is_pad = (token_emb_idx == pad_id);
  }

  float4 value;
  int tokens_idx = batch_idx * seq_len + seq_idx;
  if (is_pad) {
    if (dim_idx == 0) {
      pad_mask[tokens_idx] = 1;
    }
    value.x = 0.f;
    value.y = 0.f;
    value.z = 0.f;
    value.w = 0.f;
  } else {
    if (dim_idx == 0) {
      pad_mask[tokens_idx] = 0;
    }
    value = ((float4 *)token_emb)[token_emb_idx * hidden_dim + dim_idx];
    float4 pemb = ((float4 *)pos_emb)[seq_idx * hidden_dim + dim_idx];
    __half2 *value_h2 = (__half2 *)(&value);
    __half2 *pemb_h2 = (__half2 *)(&pemb);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 value_f2 = __half22float2(value_h2[i]);
      float2 pemb_f2 = __half22float2(pemb_h2[i]);
      value_f2.x += pemb_f2.x;
      value_f2.y += pemb_f2.y;
      value_h2[i] = __float22half2_rn(value_f2);
    }
  }
  ((float4 *)output)[idx] = value;
}

template <typename T>
void launch_enc_emb_multilg_sentence(const T *token_emb, const T *pos_emb,
                                     const int *tokens, const T *lang_emb,
                                     const int *lang_id, T *output,
                                     int *pad_mask, int pad_id, int batch_size,
                                     int seq_len, int hidden_dim,
                                     cudaStream_t stream) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  hidden_dim >>= 2;
  int nele = batch_size * seq_len * hidden_dim;
  int nblock = (nele + MAX_THREADS - 1) / MAX_THREADS;

  ker_enc_emb_multilg_sentence<T><<<nblock, MAX_THREADS, 0, stream>>>(
      token_emb, pos_emb, tokens, lang_emb, lang_id, output, pad_mask, pad_id,
      batch_size, seq_len, hidden_dim);
}

template <>
void launch_enc_emb_multilg_sentence<__half>(
    const __half *token_emb, const __half *pos_emb, const int *tokens,
    const __half *lang_emb, const int *lang_id, __half *output, int *pad_mask,
    int pad_id, int batch_size, int seq_len, int hidden_dim,
    cudaStream_t stream) {
  if (hidden_dim % 8 != 0) {
    throw std::runtime_error("violate hidden_dim % 8 = 0");
  }
  hidden_dim >>= 3;
  int nele = batch_size * seq_len * hidden_dim;
  int nblock = (nele + MAX_THREADS - 1) / MAX_THREADS;

  ker_enc_emb_multilg_sentence<__half><<<nblock, MAX_THREADS, 0, stream>>>(
      token_emb, pos_emb, tokens, lang_emb, lang_id, output, pad_mask, pad_id,
      batch_size, seq_len, hidden_dim);
}

template void launch_enc_emb_multilg_sentence<float>(
    const float *token_emb, const float *pos_emb, const int *tokens,
    const float *lang_emb, const int *lang_id, float *output, int *pad_mask,
    int pad_id, int batch_size, int seq_len, int hidden_dim,
    cudaStream_t stream);

template void launch_enc_emb_multilg_sentence<__half>(
    const __half *token_emb, const __half *pos_emb, const int *tokens,
    const __half *lang_emb, const int *lang_id, __half *output, int *pad_mask,
    int pad_id, int batch_size, int seq_len, int hidden_dim,
    cudaStream_t stream);

/**
@brief: ker_dec_embedding
for decoder, look up token embedding, add position embedding

@thread
gridDim.x = batch_size * beam_size
blockDim.x = max_thread_per_block

@param
token_emb: [hidden_size, vocab_size], note, it is different with encoder
pos_emb: [max_step, hidden_size]
token_id: input token id, [batch_size, beam_size, max_step]
output: result, [batch_size, beam_size, hidden_size]
step: current step
max_step: max decoder steps
vocab_size: vocabulary size
*/
template <typename T>
__global__ void ker_dec_embedding(const T *token_emb, const T *pos_emb,
                                  const int *token_id, T *output, int step,
                                  int max_step, int vocab_size,
                                  int hidden_size) {
  for (uint offset = threadIdx.x; offset < hidden_size; offset += blockDim.x) {
    int token_idx = token_id[blockIdx.x * max_step + step];
    output[blockIdx.x * hidden_size + offset] =
        token_emb[offset * vocab_size + token_idx] +
        pos_emb[step * hidden_size + offset];
  }
}

template <typename T>
void ker_dec_embedding_launcher(int step_token_num, int hidden_size,
                                cudaStream_t stream, const T *token_emb,
                                const T *pos_emb, const int *token_id,
                                T *output, int step, int max_step,
                                int vocab_size, int max_thread_per_block) {
  ker_dec_embedding<T><<<step_token_num, max_thread_per_block, 0, stream>>>(
      token_emb, pos_emb, token_id, output, step, max_step, vocab_size,
      hidden_size);
}

template void ker_dec_embedding_launcher<float>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const float *token_emb, const float *pos_emb, const int *token_id,
    float *output, int step, int max_step, int vocab_size,
    int max_thread_per_block);

template void ker_dec_embedding_launcher<__half>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const __half *token_emb, const __half *pos_emb, const int *token_id,
    __half *output, int step, int max_step, int vocab_size,
    int max_thread_per_block);
