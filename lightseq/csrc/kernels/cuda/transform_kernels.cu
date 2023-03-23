#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>

#include "kernels.h"

using namespace cub;

namespace lightseq {
namespace cuda {

template <typename T>
bool check_divide_float4(int sz) {
  return !(sz & 3);  // sz % 4 == 0
}

template <>
bool check_divide_float4<__half>(int sz) {
  return !(sz & 7);  // sz % 8 == 0
}

template <typename T>
void divide_float4(int *sz) {
  if ((*sz) % 4 != 0) {
    throw std::runtime_error("size need to be a multiple of 4 when use float4");
  }
  (*sz) >>= 2;
}

template <>
void divide_float4<__half>(int *sz) {
  if ((*sz) % 8 != 0) {
    throw std::runtime_error("size need to be a multiple of 8 when use float4");
  }
  (*sz) >>= 3;
}

/**
@brief: transform_0213
transform a tensor from
[sz0, sz1, sz2, sz3] to [sz0, sz2, sz1, sz3]

@thread
gridDim.x = (num_all + max_block_thread - 1) / max_block_thread
blockDim.x = max_block_thread

@param
input: [sz0, sz1, sz2, sz3]
output: [sz0, sz2, sz1, sz3]
*/
template <typename T>
__global__ void ker_transform_0213_float4(const T *input, T *output, int sz0,
                                          int sz1, int sz2, int sz3) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  int num_all = sz0 * sz1 * sz2 * sz3;
  if (offset >= num_all) {
    return;
  }
  int id0, id1, id2, id3;
  decompose_4dim(offset, sz1, sz2, sz3, &id0, &id1, &id2, &id3);
  int trg_offset = flat_4dim(id0, id2, id1, id3, sz2, sz1, sz3);
  const float4 *src = reinterpret_cast<const float4 *>(input);
  float4 *trg = reinterpret_cast<float4 *>(output);
  trg[trg_offset] = src[offset];
}

template <typename T>
__global__ void ker_transform_0213(const T *input, T *output, int sz0, int sz1,
                                   int sz2, int sz3) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  int num_all = sz0 * sz1 * sz2 * sz3;
  if (offset >= num_all) {
    return;
  }
  int id0, id1, id2, id3;
  decompose_4dim(offset, sz1, sz2, sz3, &id0, &id1, &id2, &id3);
  int trg_offset = flat_4dim(id0, id2, id1, id3, sz2, sz1, sz3);
  output[trg_offset] = input[offset];
}

//[sz0, sz1, sz2, sz3] -> [sz0, sz2, sz1, sz3]
template <typename T>
void launch_transform_0213(const T *input, T *output, int sz0, int sz1, int sz2,
                           int sz3, cudaStream_t stream) {
  if (check_divide_float4<T>(sz3)) {
    divide_float4<T>(&sz3);
    int num_all = sz0 * sz1 * sz2 * sz3;
    int nblock = (num_all + MAX_THREADS - 1) / MAX_THREADS;
    ker_transform_0213_float4<T>
        <<<nblock, MAX_THREADS, 0, stream>>>(input, output, sz0, sz1, sz2, sz3);
  } else {
    int num_all = sz0 * sz1 * sz2 * sz3;
    int nblock = (num_all + MAX_THREADS - 1) / MAX_THREADS;
    ker_transform_0213<T>
        <<<nblock, MAX_THREADS, 0, stream>>>(input, output, sz0, sz1, sz2, sz3);
  }
}

template void launch_transform_0213<float>(const float *input, float *output,
                                           int sz0, int sz1, int sz2, int sz3,
                                           cudaStream_t stream);
template void launch_transform_0213<__half>(const __half *input, __half *output,
                                            int sz0, int sz1, int sz2, int sz3,
                                            cudaStream_t stream);

/**
@brief: bias_add_transform_20314
Add bias to input, transform from
[0, 1, 2, 3, 4] to [2, 0, 3, 1, 4]

@thread
gridDim.x = dim_0
gridDim.y = dim_1
gridDim.z = dim_2
blockDim.x = min(dim_3 * dim_4, MAX_THREADS)

@param
input: [dim_0, dim_1, dim_2, dim_3, dim_4]
bias: [dim_2, dim_3, dim_4]
output: [dim_2, dim_0, dim_3, dim_1, dim_4]
*/
template <typename T>
__global__ void bias_add_transform_20314(T *output, const T *input,
                                         const T *bias, int dim_3, int dim_4);

template <>
__global__ void bias_add_transform_20314<float>(float *output,
                                                const float *input,
                                                const float *bias, int dim_3,
                                                int dim_4) {
  int id0 = blockIdx.x;
  int id1 = blockIdx.y;
  int id2 = blockIdx.z;
  int dim_0 = gridDim.x;
  int dim_1 = gridDim.y;
  int dim_2 = gridDim.z;
  int dim_34 = dim_3 * dim_4;

  int src_offset = flat_4dim(id0, id1, id2, 0, dim_1, dim_2, dim_34);
  int trg_offset = flat_5dim(id2, id0, 0, id1, 0, dim_0, dim_3, dim_1, dim_4);
  int bias_offset = flat_2dim(id2, 0, dim_34);

  const float4 *qkv4 = reinterpret_cast<const float4 *>(input);
  const float4 *bias4 = reinterpret_cast<const float4 *>(bias);
  float4 *res4 = reinterpret_cast<float4 *>(output);
  float4 vqkv4;
  float4 vbias4;
  float4 vres4;

  for (std::size_t i = threadIdx.x; i < dim_34; i += blockDim.x) {
    vqkv4 = qkv4[src_offset + i];
    vbias4 = bias4[bias_offset + i];
    vres4.x = vqkv4.x + vbias4.x;
    vres4.y = vqkv4.y + vbias4.y;
    vres4.z = vqkv4.z + vbias4.z;
    vres4.w = vqkv4.w + vbias4.w;

    int id3 = i / dim_4;
    int id4 = i % dim_4;
    int cur_trg_offset = flat_3dim(id3, 0, id4, dim_1, dim_4);
    res4[trg_offset + cur_trg_offset] = vres4;
  }
}

template <>
__global__ void bias_add_transform_20314<__half>(__half *output,
                                                 const __half *input,
                                                 const __half *bias, int dim_3,
                                                 int dim_4) {
  int id0 = blockIdx.x;
  int id1 = blockIdx.y;
  int id2 = blockIdx.z;
  int dim_0 = gridDim.x;
  int dim_1 = gridDim.y;
  int dim_2 = gridDim.z;
  int dim_34 = dim_3 * dim_4;

  int src_offset = flat_4dim(id0, id1, id2, 0, dim_1, dim_2, dim_34);
  int trg_offset = flat_5dim(id2, id0, 0, id1, 0, dim_0, dim_3, dim_1, dim_4);
  int bias_offset = flat_2dim(id2, 0, dim_34);

  const float4 *qkv4 = reinterpret_cast<const float4 *>(input);
  const float4 *bias4 = reinterpret_cast<const float4 *>(bias);
  float4 *res4 = reinterpret_cast<float4 *>(output);
  float4 vqkv4;
  float4 vbias4;
  float4 vres4;
  __half2 *h2_qkv = reinterpret_cast<__half2 *>(&vqkv4);
  __half2 *h2_bias = reinterpret_cast<__half2 *>(&vbias4);
  __half2 *h2_res = reinterpret_cast<__half2 *>(&vres4);

  for (std::size_t i = threadIdx.x; i < dim_34; i += blockDim.x) {
    vqkv4 = qkv4[src_offset + i];
    vbias4 = bias4[bias_offset + i];
    h2_res[0] = __hadd2(h2_qkv[0], h2_bias[0]);
    h2_res[1] = __hadd2(h2_qkv[1], h2_bias[1]);
    h2_res[2] = __hadd2(h2_qkv[2], h2_bias[2]);
    h2_res[3] = __hadd2(h2_qkv[3], h2_bias[3]);

    int id3 = i / dim_4;
    int id4 = i % dim_4;
    int cur_trg_offset = flat_3dim(id3, 0, id4, dim_1, dim_4);
    res4[trg_offset + cur_trg_offset] = vres4;
  }
}

// [b, s, 3, h] -> [3, b, nh, s, ad]
template <>
void launch_bias_add_transform_20314<float>(float *output, const float *input,
                                            const float *bias, int dim_0,
                                            int dim_1, int dim_2, int dim_3,
                                            int dim_4, cudaStream_t stream) {
  dim_4 >>= 2;

  dim3 grid_dim(dim_0, dim_1, dim_2);
  dim3 block_dim(min(dim_3 * dim_4, MAX_THREADS));

  bias_add_transform_20314<float>
      <<<grid_dim, block_dim, 0, stream>>>(output, input, bias, dim_3, dim_4);
}

template <>
void launch_bias_add_transform_20314<__half>(__half *output,
                                             const __half *input,
                                             const __half *bias, int dim_0,
                                             int dim_1, int dim_2, int dim_3,
                                             int dim_4, cudaStream_t stream) {
  dim_4 >>= 3;

  dim3 grid_dim(dim_0, dim_1, dim_2);
  dim3 block_dim(min(dim_3 * dim_4, MAX_THREADS));

  bias_add_transform_20314<__half>
      <<<grid_dim, block_dim, 0, stream>>>(output, input, bias, dim_3, dim_4);
}

/**
@brief: quant_bias_add_transform_20314
Add bias to input, transform from
[0, 1, 2, 3, 4] to [2, 0, 3, 1, 4]

@thread
gridDim.x = dim_0
gridDim.y = dim_1
gridDim.z = dim_2
blockDim.x = min(dim_3 * dim_4, MAX_THREADS)

@param
input: [dim_0, dim_1, dim_2, dim_3, dim_4]
bias: [dim_2, dim_3, dim_4]
output: [dim_2, dim_0, dim_3, dim_1, dim_4]
*/
template <typename T>
__global__ void quant_bias_add_transform_20314(T *output, uint8_t *clip_mask,
                                               const int8_t *input,
                                               const T *bias, const T *clip_max,
                                               int dim_3, int dim_4,
                                               const T *out_clip_max,
                                               bool in_col32);

template <>
__global__ void quant_bias_add_transform_20314<float>(
    float *output, uint8_t *clip_mask, const int8_t *input, const float *bias,
    const float *clip_max, int dim_3, int dim_4, const float *out_clip_max,
    bool in_col32) {
  int id0 = blockIdx.x;
  int id1 = blockIdx.y;
  int id2 = blockIdx.z;
  int dim_0 = gridDim.x;
  int dim_1 = gridDim.y;
  int dim_2 = gridDim.z;
  int dim_34 = dim_3 * dim_4;

  int src_offset = flat_4dim(id0, id1, id2, 0, dim_1, dim_2, dim_34);
  int trg_offset = flat_5dim(id2, id0, 0, id1, 0, dim_0, dim_3, dim_1, dim_4);
  int bias_offset = flat_2dim(id2, 0, dim_34);

  const int32_t *qkv4 = reinterpret_cast<const int32_t *>(input);
  const float4 *bias4 = reinterpret_cast<const float4 *>(bias);
  float4 *res4 = reinterpret_cast<float4 *>(output);

  int32_t vqkv4;
  float4 vbias4;
  float4 vres4;

  float clip_max_val = clip_max[0];
  float out_clip_max_val;
  if (out_clip_max) out_clip_max_val = out_clip_max[0];
  // fix me
  uint8_t clip_mask_val;

  for (std::size_t i = threadIdx.x; i < dim_34; i += blockDim.x) {
    int input_index;
    if (in_col32) {
      int idx = src_offset + i;
      int batch_tokens = dim_0 * dim_1;
      int hidden_size = dim_2 * dim_34 * 4;
      int row_id = (idx * 4) / hidden_size;
      int col_id = (idx * 4) % hidden_size;
      input_index =
          row_major2flat_col32(row_id, col_id, batch_tokens, hidden_size) / 4;
    } else {
      input_index = src_offset + i;
    }
    vqkv4 = qkv4[input_index];
    vbias4 = bias4[bias_offset + i];
    int8_t *qkv = reinterpret_cast<int8_t *>(&vqkv4);
    vres4.x = dequantize(qkv[0], clip_max_val) + vbias4.x;
    vres4.y = dequantize(qkv[1], clip_max_val) + vbias4.y;
    vres4.z = dequantize(qkv[2], clip_max_val) + vbias4.z;
    vres4.w = dequantize(qkv[3], clip_max_val) + vbias4.w;

    if (out_clip_max) {
      vres4.x = fake_quantize(vres4.x, out_clip_max_val, clip_mask_val, 6);
      vres4.y = fake_quantize(vres4.y, out_clip_max_val, clip_mask_val, 6);
      vres4.z = fake_quantize(vres4.z, out_clip_max_val, clip_mask_val, 6);
      vres4.w = fake_quantize(vres4.w, out_clip_max_val, clip_mask_val, 6);
    }

    int id3 = i / dim_4;
    int id4 = i % dim_4;
    int cur_trg_offset = flat_3dim(id3, 0, id4, dim_1, dim_4);
    res4[trg_offset + cur_trg_offset] = vres4;
  }
}

template <>
__global__ void quant_bias_add_transform_20314<__half>(
    __half *output, uint8_t *clip_mask, const int8_t *input, const __half *bias,
    const __half *clip_max, int dim_3, int dim_4, const __half *out_clip_max,
    bool in_col32) {
  int id0 = blockIdx.x;
  int id1 = blockIdx.y;
  int id2 = blockIdx.z;
  int dim_0 = gridDim.x;
  int dim_1 = gridDim.y;
  int dim_2 = gridDim.z;
  int dim_34 = dim_3 * dim_4;

  int src_offset = flat_4dim(id0, id1, id2, 0, dim_1, dim_2, dim_34);
  int trg_offset = flat_5dim(id2, id0, 0, id1, 0, dim_0, dim_3, dim_1, dim_4);
  int bias_offset = flat_2dim(id2, 0, dim_34);

  // const float4 *qkv4 = reinterpret_cast<const float4 *>(input);
  const int64_t *qkv8 = reinterpret_cast<const int64_t *>(input);
  const float4 *bias4 = reinterpret_cast<const float4 *>(bias);
  float4 *res4 = reinterpret_cast<float4 *>(output);
  int64_t vqkv8;
  float4 vbias4;
  float4 vres4;
  int8_t *qkv = reinterpret_cast<int8_t *>(&vqkv8);
  __half2 *h2_bias = reinterpret_cast<__half2 *>(&vbias4);
  __half2 *h2_res = reinterpret_cast<__half2 *>(&vres4);

  float clip_max_val = __half2float(clip_max[0]);
  float out_clip_max_val;
  if (out_clip_max) out_clip_max_val = __half2float(out_clip_max[0]);
  uint8_t clip_mask_val;

  for (std::size_t i = threadIdx.x; i < dim_34; i += blockDim.x) {
    int input_index;
    if (in_col32) {
      int idx = src_offset + i;
      int hidden_size = dim_2 * dim_34 * 8;
      int batch_tokens = dim_0 * dim_1;
      int row_id = (idx * 8) / hidden_size;
      int col_id = (idx * 8) % hidden_size;
      input_index =
          row_major2flat_col32(row_id, col_id, batch_tokens, hidden_size) / 8;
    } else {
      input_index = src_offset + i;
    }
    vqkv8 = qkv8[input_index];
    vbias4 = bias4[bias_offset + i];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      h2_res[j] =
          __hadd2(__floats2half2_rn(dequantize(qkv[j * 2], clip_max_val),
                                    dequantize(qkv[j * 2 + 1], clip_max_val)),
                  h2_bias[j]);
      if (out_clip_max) {
        h2_res[j].x = __float2half(fake_quantize(
            __half2float(h2_res[j].x), out_clip_max_val, clip_mask_val, 6));
        h2_res[j].y = __float2half(fake_quantize(
            __half2float(h2_res[j].y), out_clip_max_val, clip_mask_val, 6));
      }
    }

    int id3 = i / dim_4;
    int id4 = i % dim_4;
    int cur_trg_offset = flat_3dim(id3, 0, id4, dim_1, dim_4);
    res4[trg_offset + cur_trg_offset] = vres4;
  }
}

template <>
void launch_quant_bias_add_transform_20314<float>(
    float *output, uint8_t *clip_mask, const int8_t *input, const float *bias,
    const float *clip_max, int dim_0, int dim_1, int dim_2, int dim_3,
    int dim_4, cudaStream_t stream, const float *out_clip_max, bool in_col32) {
  dim_4 >>= 2;

  dim3 grid_dim(dim_0, dim_1, dim_2);
  dim3 block_dim(min(dim_3 * dim_4, MAX_THREADS));

  quant_bias_add_transform_20314<float><<<grid_dim, block_dim, 0, stream>>>(
      output, clip_mask, input, bias, clip_max, dim_3, dim_4, out_clip_max,
      in_col32);
}

template <>
void launch_quant_bias_add_transform_20314<__half>(
    __half *output, uint8_t *clip_mask, const int8_t *input, const __half *bias,
    const __half *clip_max, int dim_0, int dim_1, int dim_2, int dim_3,
    int dim_4, cudaStream_t stream, const __half *out_clip_max, bool in_col32) {
  dim_4 >>= 3;

  dim3 grid_dim(dim_0, dim_1, dim_2);
  dim3 block_dim(min(dim_3 * dim_4, MAX_THREADS));

  quant_bias_add_transform_20314<__half><<<grid_dim, block_dim, 0, stream>>>(
      output, clip_mask, input, bias, clip_max, dim_3, dim_4, out_clip_max,
      in_col32);
}

/**
@brief: transform4d_0213
Reshape the input matrix to merge the heads

@thread
gridDim.x = (num_all + max_block_thread - 1) / max_block_thread
blockDim.x = max_block_thread

@param
input: [trans_count, batch_size, nhead, seq_len, head_dim]
output: [batch_size, seq_len, trans_count, nhead, head_dim]
batch_size: the size of the current batch
seq_len: the sequence length of the current batch
hidden_dim: dim of the hidden tensor
nhead: number of attention heads
trans_count: 1 or 3, the count of matrice need to be transformed
*/
template <typename T>
__global__ void transform4d_0213(T *output, const T *input, int batch_size,
                                 int seq_len, int trans_count, int nhead,
                                 int head_dim, int num_all) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= num_all) {
    return;
  }
  int trans_id, batch_id, head_id, token_id, dim_id;
  decompose_5dim(offset, batch_size, nhead, seq_len, head_dim, &trans_id,
                 &batch_id, &head_id, &token_id, &dim_id);
  // [b, s, tc, nh, ad]
  int trg_offset = flat_5dim(batch_id, token_id, trans_id, head_id, dim_id,
                             seq_len, trans_count, nhead, head_dim);

  const float4 *input4 = reinterpret_cast<const float4 *>(input);
  float4 *res4 = reinterpret_cast<float4 *>(output);
  res4[trg_offset] = input4[offset];
}

/**
@brief: transform4d_0213_slow
Reshape the input matrix to merge the heads
Not use float4 for dim % 4 != 0 or dim % 8 != 0

@thread
gridDim.x = (num_all + max_block_thread - 1) / max_block_thread
blockDim.x = max_block_thread

@param
input: [trans_count, batch_size, nhead, seq_len, head_dim]
output: [batch_size, seq_len, trans_count, nhead, head_dim]
batch_size: the size of the current batch
seq_len: the sequence length of the current batch
hidden_dim: dim of the hidden tensor
nhead: number of attention heads
trans_count: 1 or 3, the count of matrice need to be transformed
*/
template <typename T>
__global__ void transform4d_0213_slow(T *output, const T *input, int batch_size,
                                      int seq_len, int trans_count, int nhead,
                                      int head_dim, int num_all) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= num_all) {
    return;
  }
  int trans_id, batch_id, head_id, token_id, dim_id;
  decompose_5dim(offset, batch_size, nhead, seq_len, head_dim, &trans_id,
                 &batch_id, &head_id, &token_id, &dim_id);
  // [b, s, tc, nh, ad]
  int trg_offset = flat_5dim(batch_id, token_id, trans_id, head_id, dim_id,
                             seq_len, trans_count, nhead, head_dim);

  output[trg_offset] = input[offset];
}

// [tc, b, nh, s, ad] -> [b, s, tc, nh, ad]
template <>
void launch_transform4d_0213<float>(float *output, const float *input,
                                    int batch_size, int seq_len, int hidden_dim,
                                    int nhead, int trans_count,
                                    cudaStream_t stream) {
  if ((hidden_dim / nhead) % 4 == 0) {
    hidden_dim >>= 2;
    int head_dim = hidden_dim / nhead;
    int num_all = batch_size * seq_len * trans_count * hidden_dim;
    int nblock = (num_all + MAX_THREADS - 1) / MAX_THREADS;

    transform4d_0213<float><<<nblock, MAX_THREADS, 0, stream>>>(
        output, input, batch_size, seq_len, trans_count, nhead, head_dim,
        num_all);
  } else {
    int head_dim = hidden_dim / nhead;
    int num_all = batch_size * seq_len * trans_count * hidden_dim;
    int nblock = (num_all + MAX_THREADS - 1) / MAX_THREADS;

    transform4d_0213_slow<float><<<nblock, MAX_THREADS, 0, stream>>>(
        output, input, batch_size, seq_len, trans_count, nhead, head_dim,
        num_all);
  }
}

template <>
void launch_transform4d_0213<__half>(__half *output, const __half *input,
                                     int batch_size, int seq_len,
                                     int hidden_dim, int nhead, int trans_count,
                                     cudaStream_t stream) {
  if ((hidden_dim / nhead) % 8 == 0) {
    hidden_dim >>= 3;
    int head_dim = hidden_dim / nhead;
    int num_all = batch_size * seq_len * trans_count * hidden_dim;
    int nblock = (num_all + MAX_THREADS - 1) / MAX_THREADS;

    transform4d_0213<__half><<<nblock, MAX_THREADS, 0, stream>>>(
        output, input, batch_size, seq_len, trans_count, nhead, head_dim,
        num_all);
  } else {
    int head_dim = hidden_dim / nhead;
    int num_all = batch_size * seq_len * trans_count * hidden_dim;
    int nblock = (num_all + MAX_THREADS - 1) / MAX_THREADS;

    transform4d_0213_slow<__half><<<nblock, MAX_THREADS, 0, stream>>>(
        output, input, batch_size, seq_len, trans_count, nhead, head_dim,
        num_all);
  }
}

/**
@brief: quant_transform4d_0213
Reshape the input matrix to merge the heads, and quantize output

@thread
gridDim.x = (num_all + max_block_thread - 1) / max_block_thread
blockDim.x = max_block_thread

@param
input: [trans_count, batch_size, nhead, seq_len, head_dim]
output: [batch_size, seq_len, trans_count, nhead, head_dim]
batch_size: the size of the current batch
seq_len: the sequence length of the current batch
hidden_dim: dim of the hidden tensor
nhead: number of attention heads
trans_count: 1 or 3, the count of matrice need to be transformed
*/
template <typename T>
__global__ void quant_transform4d_0213(int8_t *output, uint8_t *clip_mask,
                                       const T *input, const T *clip_max,
                                       int batch_size, int seq_len,
                                       int trans_count, int nhead, int head_dim,
                                       int num_all) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= num_all) {
    return;
  }
  int trans_id, batch_id, head_id, token_id, dim_id;
  decompose_5dim(offset, batch_size, nhead, seq_len, head_dim, &trans_id,
                 &batch_id, &head_id, &token_id, &dim_id);
  // [b, s, tc, nh, ad]
  int trg_offset = flat_5dim(batch_id, token_id, trans_id, head_id, dim_id,
                             seq_len, trans_count, nhead, head_dim);

  float clip_max_val = clip_max[0];

  const float4 *input4 = reinterpret_cast<const float4 *>(input);
  int8_t res[4];
  uint8_t cmask[4];
  float4 input4_i = input4[offset];
  res[0] = quantize(input4_i.x, clip_max_val, cmask[0], 2);
  res[1] = quantize(input4_i.y, clip_max_val, cmask[1], 2);
  res[2] = quantize(input4_i.z, clip_max_val, cmask[2], 2);
  res[3] = quantize(input4_i.w, clip_max_val, cmask[3], 2);

  int32_t *res4 = reinterpret_cast<int32_t *>(output);
  uint32_t *cmask4 = reinterpret_cast<uint32_t *>(clip_mask);
  res4[trg_offset] = reinterpret_cast<int32_t *>(res)[0];
  cmask4[trg_offset] |= reinterpret_cast<uint32_t *>(cmask)[0];
}

template <>
__global__ void quant_transform4d_0213<__half>(
    int8_t *output, uint8_t *clip_mask, const __half *input,
    const __half *clip_max, int batch_size, int seq_len, int trans_count,
    int nhead, int head_dim, int num_all) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= num_all) {
    return;
  }
  int trans_id, batch_id, head_id, token_id, dim_id;
  decompose_5dim(offset, batch_size, nhead, seq_len, head_dim, &trans_id,
                 &batch_id, &head_id, &token_id, &dim_id);
  // [b, s, tc, nh, ad]
  int trg_offset = flat_5dim(batch_id, token_id, trans_id, head_id, dim_id,
                             seq_len, trans_count, nhead, head_dim);

  float clip_max_val = __half2float(clip_max[0]);

  const float4 *input_f4 = reinterpret_cast<const float4 *>(input);
  int8_t res[8];
  uint8_t cmask[8];
  float4 input_f4_i = input_f4[offset];
  __half *input8 = reinterpret_cast<__half *>(&input_f4_i);
#pragma unroll
  for (int i = 0; i < 8; i++) {
    res[i] = quantize(__half2float(input8[i]), clip_max_val, cmask[i], 2);
  }

  int64_t *res8 = reinterpret_cast<int64_t *>(output);
  uint64_t *cmask8 = reinterpret_cast<uint64_t *>(clip_mask);
  res8[trg_offset] = reinterpret_cast<int64_t *>(res)[0];
  cmask8[trg_offset] |= reinterpret_cast<uint64_t *>(cmask)[0];
}

// [tc, b, nh, s, ad] -> [b, s, tc, nh, ad]
template <>
void launch_quant_transform4d_0213<float>(int8_t *output, uint8_t *clip_mask,
                                          const float *vals,
                                          const float *clip_max, int batch_size,
                                          int seq_len, int hidden_dim,
                                          int nhead, int trans_count,
                                          cudaStream_t stream) {
  hidden_dim >>= 2;
  int head_dim = hidden_dim / nhead;
  int num_all = batch_size * seq_len * trans_count * hidden_dim;
  int nblock = (num_all + MAX_THREADS - 1) / MAX_THREADS;

  quant_transform4d_0213<float><<<nblock, MAX_THREADS, 0, stream>>>(
      output, clip_mask, vals, clip_max, batch_size, seq_len, trans_count,
      nhead, head_dim, num_all);
}

template <>
void launch_quant_transform4d_0213<__half>(
    int8_t *output, uint8_t *clip_mask, const __half *vals,
    const __half *clip_max, int batch_size, int seq_len, int hidden_dim,
    int nhead, int trans_count, cudaStream_t stream) {
  hidden_dim >>= 3;
  int head_dim = hidden_dim / nhead;
  int num_all = batch_size * seq_len * trans_count * hidden_dim;
  int nblock = (num_all + MAX_THREADS - 1) / MAX_THREADS;

  quant_transform4d_0213<__half><<<nblock, MAX_THREADS, 0, stream>>>(
      output, clip_mask, vals, clip_max, batch_size, seq_len, trans_count,
      nhead, head_dim, num_all);
}

/**
@brief: transform4d_0213_dcmax
Reshape the input matrix to merge the heads, and reduce grad of clip_max

@thread
gridDim.x = (num_all + max_block_thread - 1) / max_block_thread
blockDim.x = max_block_thread

@param
input: [trans_count, batch_size, nhead, seq_len, head_dim]
output: [batch_size, seq_len, trans_count, nhead, head_dim]
batch_size: the size of the current batch
seq_len: the sequence length of the current batch
hidden_dim: dim of the hidden tensor
nhead: number of attention heads
trans_count: 1 or 3, the count of matrice need to be transformed
*/
template <typename T>
__global__ void transform_0213_dcmax(T *output, T *grad_cmax, const T *input,
                                     const uint8_t *clip_mask, int hidden_dim,
                                     int head_dim) {
  int batch_id = blockIdx.x;
  int token_id = blockIdx.y;
  int seq_len = gridDim.y;
  int nhead = hidden_dim / head_dim;

  // [b, s, h]
  int src_offset = flat_3dim(batch_id, token_id, 0, seq_len, hidden_dim);
  // [b, nh, s, ad]
  int trg_offset =
      flat_4dim(batch_id, 0, token_id, 0, nhead, seq_len, head_dim);

  const float4 *input4 = reinterpret_cast<const float4 *>(input);
  const uint32_t *cmask4 = reinterpret_cast<const uint32_t *>(clip_mask);
  float4 *res4 = reinterpret_cast<float4 *>(output);
  float4 vinput4, voutput4;
  float thread_cmax_grad = 0;
  float cmax_grad = 0;
  uint32_t cmask4_i;
  uint8_t *cmask = reinterpret_cast<uint8_t *>(&cmask4_i);

  for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
    vinput4 = input4[src_offset + i];
    cmask4_i = cmask4[src_offset + i];
    int head_id = i / head_dim;
    int dim_id = i % head_dim;
    int cur_trg_offset = flat_3dim(head_id, 0, dim_id, seq_len, head_dim);

    clip_bwd(voutput4.x, cmax_grad, vinput4.x, cmask[0], 2);
    thread_cmax_grad += cmax_grad;
    clip_bwd(voutput4.y, cmax_grad, vinput4.y, cmask[1], 2);
    thread_cmax_grad += cmax_grad;
    clip_bwd(voutput4.z, cmax_grad, vinput4.z, cmask[2], 2);
    thread_cmax_grad += cmax_grad;
    clip_bwd(voutput4.w, cmax_grad, vinput4.w, cmask[3], 2);
    thread_cmax_grad += cmax_grad;

    res4[trg_offset + cur_trg_offset] = voutput4;
  }

  __shared__ float block_cmax_grad;

  if (threadIdx.x == 0) block_cmax_grad = 0;
  __syncthreads();

  if (thread_cmax_grad != 0) {
    atomicAdd(&block_cmax_grad, thread_cmax_grad);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    if (block_cmax_grad != 0) {
      atomicAdd(&grad_cmax[0], __float2half(block_cmax_grad));
    }
  }
}

template <>
__global__ void transform_0213_dcmax<__half>(__half *output, __half *grad_cmax,
                                             const __half *input,
                                             const uint8_t *clip_mask,
                                             int hidden_dim, int head_dim) {
  int batch_id = blockIdx.x;
  int token_id = blockIdx.y;
  int seq_len = gridDim.y;
  int nhead = hidden_dim / head_dim;

  // [b, s, h]
  int src_offset = flat_3dim(batch_id, token_id, 0, seq_len, hidden_dim);
  // [b, nh, s, ad]
  int trg_offset =
      flat_4dim(batch_id, 0, token_id, 0, nhead, seq_len, head_dim);

  const float4 *input4 = reinterpret_cast<const float4 *>(input);
  const uint64_t *cmask8 = reinterpret_cast<const uint64_t *>(clip_mask);
  float4 *res4 = reinterpret_cast<float4 *>(output);
  float4 vinput4;
  __half *input8 = reinterpret_cast<__half *>(&vinput4);
  float4 res8;
  __half *res = reinterpret_cast<__half *>(&res8);
  uint64_t cmask8_i;
  uint8_t *cmask = reinterpret_cast<uint8_t *>(&cmask8_i);
  float thread_cmax_grad = 0;
  float cmax_grad = 0;

  for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
    vinput4 = input4[src_offset + i];
    cmask8_i = cmask8[src_offset + i];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      clip_bwd(res[j], cmax_grad, input8[j], cmask[j], 2);
      thread_cmax_grad += cmax_grad;
    }

    int head_id = i / head_dim;
    int dim_id = i % head_dim;
    int cur_trg_offset = flat_3dim(head_id, 0, dim_id, seq_len, head_dim);
    res4[trg_offset + cur_trg_offset] = vinput4;
  }

  __shared__ float block_cmax_grad;

  if (threadIdx.x == 0) block_cmax_grad = 0;
  __syncthreads();

  if (thread_cmax_grad != 0) {
    atomicAdd(&block_cmax_grad, thread_cmax_grad);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    if (block_cmax_grad != 0) {
      atomicAdd(&grad_cmax[0], __float2half(block_cmax_grad));
    }
  }
}

// [b, nh, s, ad] -> [b, s, nh, ad]
template <>
void launch_transform_0213_dcmax<float>(float *output, float *grad_cmax,
                                        const float *input,
                                        const uint8_t *clip_mask,
                                        int batch_size, int seq_len,
                                        int hidden_dim, int nhead,
                                        cudaStream_t stream) {
  hidden_dim >>= 2;
  int head_dim = hidden_dim / nhead;

  dim3 grid_dim(batch_size, seq_len);
  dim3 block_dim(min(hidden_dim, MAX_THREADS));

  transform_0213_dcmax<float><<<grid_dim, block_dim, 0, stream>>>(
      output, grad_cmax, input, clip_mask, hidden_dim, head_dim);
}

template <>
void launch_transform_0213_dcmax<__half>(__half *output, __half *grad_cmax,
                                         const __half *input,
                                         const uint8_t *clip_mask,
                                         int batch_size, int seq_len,
                                         int hidden_dim, int nhead,
                                         cudaStream_t stream) {
  hidden_dim >>= 3;
  int head_dim = hidden_dim / nhead;

  dim3 grid_dim(batch_size, seq_len);
  dim3 block_dim(min(hidden_dim, MAX_THREADS));

  transform_0213_dcmax<__half><<<grid_dim, block_dim, 0, stream>>>(
      output, grad_cmax, input, clip_mask, hidden_dim, head_dim);
}

template <typename T>
__device__ void add_float4(float4 *a, float4 *b) {
  a[0].x += b[0].x;
  a[0].y += b[0].y;
  a[0].z += b[0].z;
  a[0].w += b[0].w;
}

template <>
__device__ void add_float4<__half>(float4 *a, float4 *b) {
  __half2 *a_h2 = reinterpret_cast<__half2 *>(a);
  __half2 *b_h2 = reinterpret_cast<__half2 *>(b);
  ;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 a_f2 = __half22float2(a_h2[i]);
    float2 b_f2 = __half22float2(b_h2[i]);
    a_f2.x += b_f2.x;
    a_f2.y += b_f2.y;
    a_h2[i] = __float22half2_rn(a_f2);
  }
}

/**
@brief: ker_split_head
add bias to input,
and split it into query, key, value
@thread
gridDim.x = (num_all + max_block_thread - 1) / max_block_thread
blockDim.x = max_block_thread
@param
input: [batch_size, q_len, qkv_num, hidden_size]
  qkv_num = 1 or 3, 1 for enc-dec cross attn, 3 for other attn
bias: [1, 1, qkv_num, hidden_size]
query: [batch_size, nhead, q_len, head_dim]
key: [batch_size, nhead, cache_sz, head_dim]
value: [batch_size, nhead, cache_sz, head_dim]
let's explain the SplitHeadOp by PyTorch:
input = input + bias
if qkv_num == 3:
  q, k, v = input.split(1, dim=2)
if qkv_num == 1:
  q = input
lambda func = x: x.squeeze().reshape((batch_size, seq_len,
  nhead, head_dim)).permute(0, 2, 1, 3)
query = func(q)
if qkv_num == 3:
  key[:,:,step:step+q_len,:] = func(k)
  value[:,:,step:step+q_len,:] = func(v)
*/
template <typename T>
__global__ void ker_split_head_float4(const T *inp, const T *bias, T *query,
                                      T *key, T *value, int batch_size,
                                      int hidden_dim, int head_dim, int q_len,
                                      int kv_len, int step, int qkv_num,
                                      int num_all) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= num_all) {
    return;
  }
  int nhead = hidden_dim / head_dim;
  int batch_id, token_id, qkv_id, head_id, dim_id;
  decompose_5dim(offset, q_len, qkv_num, nhead, head_dim, &batch_id, &token_id,
                 &qkv_id, &head_id, &dim_id);
  int bias_id = flat_3dim(qkv_id, head_id, dim_id, nhead, head_dim);

  float4 res4 = (reinterpret_cast<const float4 *>(inp))[offset];
  float4 bias4 = (reinterpret_cast<const float4 *>(bias))[bias_id];
  add_float4<T>(&res4, &bias4);

  float4 *trg;
  if (qkv_id == 0) {
    trg = reinterpret_cast<float4 *>(query);
    trg +=
        flat_4dim(batch_id, head_id, token_id, dim_id, nhead, q_len, head_dim);
  } else {
    if (qkv_id == 1) {
      trg = reinterpret_cast<float4 *>(key);
    } else {
      trg = reinterpret_cast<float4 *>(value);
    }
    trg += flat_4dim(batch_id, head_id, token_id + step, dim_id, nhead, kv_len,
                     head_dim);
  }
  *trg = res4;
}

template <typename T>
__global__ void ker_split_head(const T *inp, const T *bias, T *query, T *key,
                               T *value, int batch_size, int hidden_dim,
                               int head_dim, int q_len, int kv_len, int step,
                               int qkv_num, int num_all) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= num_all) {
    return;
  }
  int nhead = hidden_dim / head_dim;
  int batch_id, token_id, qkv_id, head_id, dim_id;
  decompose_5dim(offset, q_len, qkv_num, nhead, head_dim, &batch_id, &token_id,
                 &qkv_id, &head_id, &dim_id);
  int bias_id = flat_3dim(qkv_id, head_id, dim_id, nhead, head_dim);

  T *trg;
  if (qkv_id == 0) {
    trg = query;
    trg +=
        flat_4dim(batch_id, head_id, token_id, dim_id, nhead, q_len, head_dim);
  } else {
    if (qkv_id == 1) {
      trg = key;
    } else {
      trg = value;
    }
    trg += flat_4dim(batch_id, head_id, token_id + step, dim_id, nhead, kv_len,
                     head_dim);
  }
  *trg = inp[offset] + bias[bias_id];
}

template <typename T>
void launch_split_head(const T *inp, const T *bias, T *query, T *key, T *value,
                       int batch_size, int hidden_dim, int head_dim, int q_len,
                       int kv_len, int step, int qkv_num, cudaStream_t stream) {
  if (check_divide_float4<T>(hidden_dim) && check_divide_float4<T>(head_dim)) {
    divide_float4<T>(&hidden_dim);
    divide_float4<T>(&head_dim);
    int num_all = batch_size * q_len * qkv_num * hidden_dim;
    int nblock = (num_all + MAX_THREADS - 1) / MAX_THREADS;
    ker_split_head_float4<T><<<nblock, MAX_THREADS, 0, stream>>>(
        inp, bias, query, key, value, batch_size, hidden_dim, head_dim, q_len,
        kv_len, step, qkv_num, num_all);
  } else {
    int num_all = batch_size * q_len * qkv_num * hidden_dim;
    int nblock = (num_all + MAX_THREADS - 1) / MAX_THREADS;
    ker_split_head<T><<<nblock, MAX_THREADS, 0, stream>>>(
        inp, bias, query, key, value, batch_size, hidden_dim, head_dim, q_len,
        kv_len, step, qkv_num, num_all);
  }
}

template void launch_split_head<float>(const float *inp, const float *bias,
                                       float *query, float *key, float *value,
                                       int batch_size, int hidden_dim,
                                       int head_dim, int q_len, int kv_len,
                                       int step, int qkv_num,
                                       cudaStream_t stream);

template void launch_split_head<__half>(const __half *inp, const __half *bias,
                                        __half *query, __half *key,
                                        __half *value, int batch_size,
                                        int hidden_dim, int head_dim, int q_len,
                                        int kv_len, int step, int qkv_num,
                                        cudaStream_t stream);
}  // namespace cuda
}  // namespace lightseq
