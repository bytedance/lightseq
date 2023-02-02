#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>

#include "kernels.h"
#include "cstdio"

using namespace cub;

namespace lightseq {
namespace cuda {
/**
@brief: bias_add_transform_20314_new
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
__global__ void bias_add_transform_20314_new(T *q_out, T *k_out, T *v_out,
                                             const T *input, const T *bias,
                                             int dim_3, int dim_4,
                                             int batch_ele);

template <>
__global__ void bias_add_transform_20314_new<float>(
    float *q_out, float *k_out, float *v_out, const float *input,
    const float *bias, int dim_3, int dim_4, int batch_ele) {
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
  float4 *qres_4 = reinterpret_cast<float4 *>(q_out);
  float4 *kres_4 = reinterpret_cast<float4 *>(k_out);
  float4 *vres_4 = reinterpret_cast<float4 *>(v_out);

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
    int temp_offset = trg_offset + cur_trg_offset;
    if (temp_offset >= batch_ele * 2) {
      vres_4[temp_offset - batch_ele * 2] = vres4;
    } else if (temp_offset >= batch_ele) {
      kres_4[temp_offset - batch_ele] = vres4;
    } else {
      qres_4[temp_offset] = vres4;
    }
  }
}

template <>
__global__ void bias_add_transform_20314_new<__half>(
    __half *q_out, __half *k_out, __half *v_out, const __half *input,
    const __half *bias, int dim_3, int dim_4, int batch_ele) {
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
  float4 *qres_4 = reinterpret_cast<float4 *>(q_out);
  float4 *kres_4 = reinterpret_cast<float4 *>(k_out);
  float4 *vres_4 = reinterpret_cast<float4 *>(v_out);

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
    int temp_offset = trg_offset + cur_trg_offset;
    if (temp_offset >= batch_ele * 2) {
      vres_4[temp_offset - batch_ele * 2] = vres4;
    } else if (temp_offset >= batch_ele) {
      kres_4[temp_offset - batch_ele] = vres4;
    } else {
      qres_4[temp_offset] = vres4;
    }
  }
}

/**
@brief: bias_add_transform_20314_new_slow
Add bias to input, transform from
[0, 1, 2, 3, 4] to [2, 0, 3, 1, 4]
Not use float4 for dim % 4 != 0 or dim % 8 != 0

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
__global__ void bias_add_transform_20314_new_slow(T *q_out, T *k_out, T *v_out,
                                                  const T *input, const T *bias,
                                                  int dim_3, int dim_4,
                                                  int batch_ele) {
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

  float vres;

  for (std::size_t i = threadIdx.x; i < dim_34; i += blockDim.x) {
    vres = input[src_offset + i] + bias[bias_offset + i];

    int id3 = i / dim_4;
    int id4 = i % dim_4;
    int cur_trg_offset = flat_3dim(id3, 0, id4, dim_1, dim_4);
    int temp_offset = trg_offset + cur_trg_offset;
    if (temp_offset >= batch_ele * 2) {
      v_out[temp_offset - batch_ele * 2] = vres;
    } else if (temp_offset >= batch_ele) {
      k_out[temp_offset - batch_ele] = vres;
    } else {
      q_out[temp_offset] = vres;
    }
  }
}

// [b, s, 3, h] -> [3, b, nh, s, ad]
template <>
void launch_bias_add_transform_20314_new<float>(
    float *q_out, float *k_out, float *v_out, const float *input,
    const float *bias, int dim_0, int dim_1, int dim_2, int dim_3, int dim_4,
    cudaStream_t stream) {
  if (dim_4 % 4 == 0) {
    dim_4 >>= 2;

    dim3 grid_dim(dim_0, dim_1, dim_2);
    dim3 block_dim(min(dim_3 * dim_4, MAX_THREADS));
    int batch_ele = dim_0 * dim_1 * dim_3 * dim_4;

    bias_add_transform_20314_new<float><<<grid_dim, block_dim, 0, stream>>>(
        q_out, k_out, v_out, input, bias, dim_3, dim_4, batch_ele);
  } else {
    dim3 grid_dim(dim_0, dim_1, dim_2);
    dim3 block_dim(min(dim_3 * dim_4, MAX_THREADS));
    int batch_ele = dim_0 * dim_1 * dim_3 * dim_4;

    bias_add_transform_20314_new_slow<float>
        <<<grid_dim, block_dim, 0, stream>>>(q_out, k_out, v_out, input, bias,
                                             dim_3, dim_4, batch_ele);
  }
}

template <>
void launch_bias_add_transform_20314_new<__half>(
    __half *q_out, __half *k_out, __half *v_out, const __half *input,
    const __half *bias, int dim_0, int dim_1, int dim_2, int dim_3, int dim_4,
    cudaStream_t stream) {
  if (dim_4 % 8 == 0) {
    dim_4 >>= 3;

    dim3 grid_dim(dim_0, dim_1, dim_2);
    dim3 block_dim(min(dim_3 * dim_4, MAX_THREADS));

    int batch_ele = dim_0 * dim_1 * dim_3 * dim_4;

    bias_add_transform_20314_new<__half><<<grid_dim, block_dim, 0, stream>>>(
        q_out, k_out, v_out, input, bias, dim_3, dim_4, batch_ele);
  } else {
    dim3 grid_dim(dim_0, dim_1, dim_2);
    dim3 block_dim(min(dim_3 * dim_4, MAX_THREADS));

    int batch_ele = dim_0 * dim_1 * dim_3 * dim_4;

    bias_add_transform_20314_new_slow<__half>
        <<<grid_dim, block_dim, 0, stream>>>(q_out, k_out, v_out, input, bias,
                                             dim_3, dim_4, batch_ele);
  }
}

/**
@brief: transform_20314_bwd_new
Reshape the input matrix to merge the heads

@thread
gridDim.x = (batch_ele_num * 3 + max_block_thread - 1) / max_block_thread
blockDim.x = max_block_thread

@param
input: [trans_count, batch_size, nhead, seq_len, head_dim]
output: [batch_size, seq_len, trans_count, nhead, head_dim]
batch_size: the size of the current batch
seq_len: the sequence length of the current batch
hidden_dim: dim of the hidden tensor
nhead: number of attention heads
*/
template <typename T>
__global__ void transform_20314_bwd_new(T *output, const T *q_inp,
                                        const T *k_inp, const T *v_inp,
                                        int batch_size, int seq_len,
                                        int trans_count, int nhead,
                                        int head_dim, int batch_ele_num) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= batch_ele_num * trans_count) {
    return;
  }
  int trans_id, batch_id, head_id, token_id, dim_id;
  decompose_5dim(offset, batch_size, nhead, seq_len, head_dim, &trans_id,
                 &batch_id, &head_id, &token_id, &dim_id);
  // [b, s, tc, nh, ad]
  int trg_offset = flat_5dim(batch_id, token_id, trans_id, head_id, dim_id,
                             seq_len, trans_count, nhead, head_dim);

  const float4 *q_inp4 = reinterpret_cast<const float4 *>(q_inp);
  const float4 *k_inp4 = reinterpret_cast<const float4 *>(k_inp);
  const float4 *v_inp4 = reinterpret_cast<const float4 *>(v_inp);

  float4 *res4 = reinterpret_cast<float4 *>(output);
  if (offset >= batch_ele_num * 2)
    res4[trg_offset] = v_inp4[offset - batch_ele_num * 2];
  else if (offset >= batch_ele_num)
    res4[trg_offset] = k_inp4[offset - batch_ele_num];
  else
    res4[trg_offset] = q_inp4[offset];
}
}  // namespace cuda
}  // namespace lightseq
