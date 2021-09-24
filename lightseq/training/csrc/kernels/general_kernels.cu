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
gridDim.x = batch_size
gridDim.y = batch_seq_len
blockDim.x = max_thread_per_block

@param
token_emb: [vocab_size, hidden_size]
pos_emb: [max_step, hidden_size]
token_id: input token id, [batch_size, batch_seq_len]
output: result, [batch_size, batch_seq_len, hidden_size]
padding_mask: record the padding token, [batch_size, batch_seq_len]
padding_id, the padding token id
*/
template <typename T>
__global__ void ker_enc_emb(const T *token_emb, const T *pos_emb,
                            const int *token_id, T *output, int *padding_mask,
                            int padding_id, const int hidden_size) {
  int target_pos = blockIdx.x * gridDim.y + blockIdx.y;
  int start = target_pos * hidden_size + threadIdx.x;
  int end = (target_pos + 1) * hidden_size;
  int tid = token_id[target_pos];
  if (tid == padding_id) {
    // for padding id
    if (threadIdx.x == 0) padding_mask[target_pos] = 1;
    for (uint i = start; i < end; i += blockDim.x) {
      // output[target_pos * blockDim.x + threadIdx.x] = 0.f;
      output[i] = 0.f;
    }
    return;
  }
  if (threadIdx.x == 0) {
    padding_mask[target_pos] = 0;
  }
  for (uint i = start; i < end; i += blockDim.x) {
    int offset = i - target_pos * hidden_size;
    output[i] = token_emb[tid * hidden_size + offset] +
                pos_emb[blockIdx.y * hidden_size + offset];
  }
}

template <>
__global__ void ker_enc_emb<__half>(const __half *token_emb,
                                    const __half *pos_emb, const int *token_id,
                                    __half *output, int *padding_mask,
                                    int padding_id,
                                    const int half_hidden_size) {
  int target_pos = blockIdx.x * gridDim.y + blockIdx.y;
  int start = target_pos * half_hidden_size + threadIdx.x;
  int end = (target_pos + 1) * half_hidden_size;
  int tid = token_id[target_pos];
  half2 *output_h = (half2 *)output;

  if (tid == padding_id) {
    // for padding id
    if (threadIdx.x == 0) padding_mask[target_pos] = 1;
    for (uint i = start; i < end; i += blockDim.x) {
      output_h[i] = __float2half2_rn(0.f);
    }
    return;
  }
  if (threadIdx.x == 0) {
    padding_mask[target_pos] = 0;
  }
  for (uint i = start; i < end; i += blockDim.x) {
    int offset = i - target_pos * half_hidden_size;
    float2 te = __half22float2(
        ((const half2 *)token_emb)[tid * half_hidden_size + offset]);
    float2 pe = __half22float2(
        ((const half2 *)pos_emb)[blockIdx.y * half_hidden_size + offset]);
    te.x += pe.x;
    te.y += pe.y;
    output_h[i] = __float22half2_rn(te);
  }
}

template <typename T>
void launch_enc_emb(int batch_size, int batch_seq_len, int hidden_size,
                    cudaStream_t stream, const T *token_emb, const T *pos_emb,
                    const int *token_id, T *output, int *padding_mask,
                    int padding_id, int max_thread_per_block) {
  ker_enc_emb<T>
      <<<dim3(batch_size, batch_seq_len), max_thread_per_block, 0, stream>>>(
          token_emb, pos_emb, token_id, output, padding_mask, padding_id,
          hidden_size);
}

template <>
void launch_enc_emb<__half>(int batch_size, int batch_seq_len, int hidden_size,
                            cudaStream_t stream, const __half *token_emb,
                            const __half *pos_emb, const int *token_id,
                            __half *output, int *padding_mask, int padding_id,
                            int max_thread_per_block) {
  ker_enc_emb<__half>
      <<<dim3(batch_size, batch_seq_len), max_thread_per_block, 0, stream>>>(
          token_emb, pos_emb, token_id, output, padding_mask, padding_id,
          hidden_size / 2);
}

template void launch_enc_emb<float>(int batch_size, int batch_seq_len,
                                    int hidden_size, cudaStream_t stream,
                                    const float *token_emb,
                                    const float *pos_emb, const int *token_id,
                                    float *output, int *padding_mask,
                                    int padding_id, int max_thread_per_block);

template void launch_enc_emb<__half>(int batch_size, int batch_seq_len,
                                     int hidden_size, cudaStream_t stream,
                                     const __half *token_emb,
                                     const __half *pos_emb, const int *token_id,
                                     __half *output, int *padding_mask,
                                     int padding_id, int max_thread_per_block);
