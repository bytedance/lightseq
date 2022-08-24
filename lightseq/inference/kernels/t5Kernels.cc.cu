#include "t5Kernels.h"
#include "common.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

/**
@file
Implemented the cuda kernel function and its launcher
that required by transformer model.
Currently, fp16 and fp32 versions are provided
*/
namespace lightseq {
namespace cuda {
/**
@brief: t5_ker_norm_layer
layer normalization

@thread
gridDim.x = batch_size * batch_seq_len
blockDim.x = max_thread_per_block

@param
matrix: [batch_size, batch_seq_len, hidden_size]
scale: [hidden_size]
bias: [hidden_size]
*/
template <typename T>
__global__ void t5_ker_norm_layer(T* matrix, T* out, const T* scale,
                                  const T* bias, int hidden_size) {
  uint block_start = blockIdx.x * hidden_size;
  uint start = block_start + threadIdx.x;
  uint end = block_start + hidden_size;

  float val = 0.0;
  // step 1. compute variance
  for (uint i = start; i < end; i += blockDim.x) {
    float tmp = matrix[i];  // - s_mean;
    val += tmp * tmp;
  }
  __shared__ float s_var;
  float reduce_res = blockReduceSum(val);
  if (threadIdx.x == 0) {
    s_var = rsqrtf(reduce_res / float(hidden_size) + t5_epsilon);
  }
  __syncthreads();

  // step 2. layer norm
  for (uint i = start; i < end; i += blockDim.x) {
    out[i] = matrix[i] * s_var * __ldg(&scale[i - block_start]) +
             __ldg(&bias[i - block_start]);
  }
}

template <>
__global__ void t5_ker_norm_layer<__half>(__half* matrix, __half* out,
                                          const __half* scale,
                                          const __half* bias,
                                          int half_hidden_size) {
  uint block_start = blockIdx.x * half_hidden_size;
  uint start = block_start + threadIdx.x;
  uint end = blockIdx.x * half_hidden_size + half_hidden_size;
  half2* pmatrix = (half2*)matrix;
  half2* pout = (half2*)out;
  const half2* pscale = (const half2*)scale;
  const half2* pbias = (const half2*)bias;
  float mean_dim = float(half_hidden_size) * 2.f;

  float val = 0.0;

  // step 1. compute variance
  val = 0.0;
  for (uint i = start; i < end; i += blockDim.x) {
    float2 local_f2 = safe_half2_to_float2(pmatrix[i]);
    float tmpx = local_f2.x;
    float tmpy = local_f2.y;
    val += tmpx * tmpx + tmpy * tmpy;
  }
  __shared__ float s_var;
  float reduce_res = blockReduceSum(val);
  if (threadIdx.x == 0) s_var = rsqrtf(reduce_res / mean_dim + t5_epsilon);

  __syncthreads();

  // step 2. layer norm
  for (uint i = start; i < end; i += blockDim.x) {
    float2 scale_val = __half22float2(__ldg(&pscale[i - block_start]));
    float2 bias_val = __half22float2(__ldg(&pbias[i - block_start]));
    float2 local_f2 = safe_half2_to_float2(pmatrix[i]);
    local_f2.x = local_f2.x * s_var * scale_val.x + bias_val.x;
    local_f2.y = local_f2.y * s_var * scale_val.y + bias_val.y;
    pout[i] = __float22half2_rn(local_f2);
  }
}

template <typename T>
void t5_ker_norm_layer_launcher(int token_num, int hidden_size,
                                cudaStream_t stream, T* matrix, T* out,
                                const T* scale, const T* bias,
                                int max_thread_per_block) {
  t5_ker_norm_layer<T><<<token_num, max_thread_per_block, 0, stream>>>(
      matrix, out, scale, bias, hidden_size);
}

template <>
void t5_ker_norm_layer_launcher<__half>(int token_num, int hidden_size,
                                        cudaStream_t stream, __half* matrix,
                                        __half* out, const __half* scale,
                                        const __half* bias,
                                        int max_thread_per_block) {
  t5_ker_norm_layer<__half><<<token_num, max_thread_per_block, 0, stream>>>(
      matrix, out, scale, bias, hidden_size / 2);
}

__device__ int get_bucket_num(int row, int col, bool bidirectional,
                              int num_buckets = 32, int max_distance = 128) {
  int relative_position = col - row;
  int relative_buckets = 0;
  if (bidirectional) {
    num_buckets /= 2;
    if (relative_position > 0) relative_buckets += num_buckets;
    relative_position = abs(relative_position);
  } else
    relative_position = -min(relative_position, 0);

  int max_exact = num_buckets / 2;
  int relative_position_if_large =
      max_exact +
      (log((double)relative_position / max_exact) /
       log((double)max_distance / max_exact) * (num_buckets - max_exact));

  if (relative_position < max_exact)
    relative_buckets += relative_position;
  else
    relative_buckets += relative_position_if_large;
  return relative_buckets;
}

/**
@brief: t5_ker_correlation_softmax_encself
query-key correlation softmax for encoder self attention

@thread
gridDim.x = batch_size
gridDim.y = head_num * batch_seq_len
blockDim.x = first multiple of WARP_SIZE greater than batch_seq_len

@param
correlation: [batch_size, head_num, batch_seq_len, batch_seq_len]
src_padding_mask: [batch_size, batch_seq_len],
  indicating which token is a padding token.
*/
template <typename T>
__global__ void t5_ker_correlation_softmax_encself(T* correlation,
                                                   const int* src_padding_mask,
                                                   int batch_seq_len,
                                                   const T* pos_emb,
                                                   int head_num) {
  int idx = (blockIdx.x * gridDim.y + blockIdx.y) * batch_seq_len + threadIdx.x;
  if (threadIdx.x < batch_seq_len &&
      src_padding_mask[blockIdx.x * batch_seq_len +
                       blockIdx.y % batch_seq_len]) {
    correlation[idx] = (T)0.f;
    return;
  }
  int mask = threadIdx.x < batch_seq_len
                 ? src_padding_mask[blockIdx.x * batch_seq_len + threadIdx.x]
                 : 1;

  float val;
  if (threadIdx.x < batch_seq_len) {
    // We know that idx = head_index * batch_seq_len * batch_seq_len
    //     + i * batch_seq_len + j + batch_num * head_num * batch_seq_len *
    //     batch_seq_len;
    int need = idx % (head_num * batch_seq_len * batch_seq_len);
    // need = head_index * batch_seq_len * batch_seq_len + i * batch_seq_len +
    // j;
    int j = need % batch_seq_len;
    int i = (need - j) / batch_seq_len % batch_seq_len;
    int head_idx =
        (need - j - i * batch_seq_len) / batch_seq_len / batch_seq_len;
    val = (float)correlation[idx];
    // new_values[0, head, i, j] =
    // relative_attention_bias.weight[relative_position_bucket[i][j]][head]
    int bucket_index = get_bucket_num(i, j, true);
    val += (float)pos_emb[bucket_index * head_num + head_idx];
  } else
    val = CUDA_FLOAT_INF_NEG;

  float max_val = blockReduceMax<float>(mask ? CUDA_FLOAT_INF_NEG : val);
  __shared__ float smax;
  if (threadIdx.x == 0) smax = max_val;
  __syncthreads();

  val = mask ? 0.f : expf(val - smax);
  float rsum = blockReduceSum<float>(val);
  __shared__ float ssum;
  if (threadIdx.x == 0) ssum = rsum;
  __syncthreads();

  if (threadIdx.x < batch_seq_len) correlation[idx] = (T)(val / ssum);
}

template <typename T>
void t5_ker_correlation_softmax_encself_launcher(
    int batch_size, int batch_seq_len, int head_num, cudaStream_t stream,
    T* correlation, const int* src_padding_mask, const T* pos_emb) {
  int block_dim = batch_seq_len;
  if (batch_seq_len < 1024) {
    block_dim = (batch_seq_len + 31) >> 5;
    block_dim *= 32;
  }

  t5_ker_correlation_softmax_encself<T>
      <<<dim3(batch_size, head_num * batch_seq_len), block_dim, 0, stream>>>(
          correlation, src_padding_mask, batch_seq_len, pos_emb, head_num);
}

template void t5_ker_correlation_softmax_encself_launcher<float>(
    int batch_size, int batch_seq_len, int head_num, cudaStream_t stream,
    float* correlation, const int* src_padding_mask, const float* pos_emb);

template void t5_ker_correlation_softmax_encself_launcher<__half>(
    int batch_size, int batch_seq_len, int head_num, cudaStream_t stream,
    __half* correlation, const int* src_padding_mask, const __half* pos_emb);

template void t5_ker_norm_layer_launcher<float>(int token_num, int hidden_size,
                                                cudaStream_t stream,
                                                float* matrix, float* out,
                                                const float* scale,
                                                const float* bias,
                                                int max_thread_per_block);

template void t5_ker_norm_layer_launcher<__half>(int token_num, int hidden_size,
                                                 cudaStream_t stream,
                                                 __half* matrix, __half* out,
                                                 const __half* scale,
                                                 const __half* bias,
                                                 int max_thread_per_block);

/**
@brief: t5_ker_correlation_softmax_decself
query-key correlation softmax for decoder self attention

@thread
gridDim.x = batch_size * beam_size * head_num
blockDim.x = first multiple of WARP_SIZE greater than cur_step + 1

@param
correlation: [batch_size, beam_size, head_num, cur_step + 1]
*/
template <typename T>
__global__ void t5_ker_correlation_softmax_decself(T* correlation, int step_num,
                                                   const T* pos_emb,
                                                   int head_num) {
  int idx = blockIdx.x * step_num + threadIdx.x;

  float val;
  if (threadIdx.x < step_num) {
    int j = threadIdx.x;
    int i = step_num - 1;
    int head_idx = blockIdx.x % head_num;
    val = (float)correlation[idx];
    int bucket_index = get_bucket_num(i, j, false);
    val += (float)pos_emb[bucket_index * head_num + head_idx];
  } else
    val = CUDA_FLOAT_INF_NEG;

  float max_val = blockReduceMax(val);
  __shared__ float smax;
  if (threadIdx.x == 0) smax = max_val;
  __syncthreads();

  val = threadIdx.x < step_num ? expf(val - smax) : 0;

  float rsum = blockReduceSum(val);
  __shared__ float ssum;
  if (threadIdx.x == 0) ssum = rsum;
  __syncthreads();

  if (threadIdx.x < step_num) correlation[idx] = (T)(val / ssum);
}

template <typename T>
void t5_ker_correlation_softmax_decself_launcher(
    int batch_head_num, int step_num, cudaStream_t stream, T* correlation,
    const T* pos_emb, int head_num) {
  int block_dim = step_num;
  if (step_num < 1024) {
    block_dim = (step_num + 31) >> 5;
    block_dim *= 32;
  }
  t5_ker_correlation_softmax_decself<<<batch_head_num, block_dim, 0, stream>>>(
      correlation, step_num, pos_emb, head_num);
}

template void t5_ker_correlation_softmax_decself_launcher<float>(
    int batch_head_num, int step_num, cudaStream_t stream, float* correlation,
    const float* pos_emb, int head_num);

template void t5_ker_correlation_softmax_decself_launcher<__half>(
    int batch_head_num, int step_num, cudaStream_t stream, __half* correlation,
    const __half* pos_emb, int head_num);

/**
@brief: ker_gelu_first_elementmul
input activated by gelu, then element-wise add to input2, result save to input

@thread
gridDim.x = batch_size * batch_seq_len
blockDim.x = max_thread_per_block

@param
input: [batch_size * batch_seq_len, feature_dim]
bias: [feature_dim]
feature_dim: the dim of input feature
*/
template <typename T>
__global__ void ker_gelu_first_elementmul(T* input, const T* input2,
                                          int feature_dim) {
  int offset = blockIdx.x * feature_dim;
  for (int idx = threadIdx.x; idx < feature_dim; idx += blockDim.x) {
    int cur_offset = offset + idx;
    input[cur_offset] = gelu<float>(input[cur_offset]) * input2[cur_offset];
  }
}

/* fp16 version */
template <>
__global__ void ker_gelu_first_elementmul<__half>(__half* input,
                                                  const __half* input2,
                                                  int feature_dim) {
  int offset = blockIdx.x * feature_dim;
  half2* pinput = (half2*)input;
  const half2* pinput2 = (const half2*)input2;
  for (int idx = threadIdx.x; idx < feature_dim; idx += blockDim.x) {
    int cur_offset = offset + idx;
    pinput[cur_offset] =
        __hmul2(gelu<half2>(pinput[cur_offset]), pinput2[cur_offset]);
  }
}

template <typename T>
void ker_gelu_first_elementmul_launcher(int batch_token_num, int block_dim,
                                        cudaStream_t stream, T* input,
                                        const T* input2, int feature_dim) {
  ker_gelu_first_elementmul<T>
      <<<batch_token_num, block_dim, 0, stream>>>(input, input2, feature_dim);
}

template <>
void ker_gelu_first_elementmul_launcher<__half>(
    int batch_token_num, int block_dim, cudaStream_t stream, __half* input,
    const __half* input2, int feature_dim) {
  ker_gelu_first_elementmul<__half><<<batch_token_num, block_dim, 0, stream>>>(
      input, input2, feature_dim / 2);
}

template void ker_gelu_first_elementmul_launcher<float>(
    int batch_token_num, int block_dim, cudaStream_t stream, float* input,
    const float* input2, int feature_dim);

template void ker_gelu_first_elementmul_launcher<__half>(
    int batch_token_num, int block_dim, cudaStream_t stream, __half* input,
    const __half* input2, int feature_dim);

}  // namespace cuda
}  // namespace lightseq
