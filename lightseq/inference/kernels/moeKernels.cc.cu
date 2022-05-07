#include "moeKernels.h"
#include "transformerKernels.h"
#include "common.h"

/**
@file
Implemented the cuda kernel function and its launcher
that required by moe model.
Currently, fp16 and fp32 versions are provided
*/
namespace lightseq {
namespace cuda {

/**
@brief: ker_norm_layer_prepost
layer normalization, modify input according to is_post_ln

@thread
gridDim.x = batch_size * batch_seq_len
blockDim.x = max_thread_per_block

@param
input: [batch_size, batch_seq_len, hidden_size]
output: [batch_size, batch_seq_len, hidden_size]
scale: [hidden_size]
bias: [hidden_size]
*/
template <typename T>
__global__ void ker_norm_layer_prepost(T* input, T* output, const T* scale,
                                       const T* bias, const int hidden_size,
                                       bool is_post_ln) {
  uint block_start = blockIdx.x * hidden_size;
  uint start = block_start + threadIdx.x;
  uint end = block_start + hidden_size;
  float val = 0.0;
  for (uint i = start; i < end; i += blockDim.x) {
    val += input[i];
  }

  // step 0. compute mean
  __shared__ float s_mean;
  float reduce_res = blockReduceSum<float>(val);
  if (threadIdx.x == 0) s_mean = reduce_res / float(hidden_size);
  __syncthreads();

  // step 1. compute variance
  val = 0.0;
  for (uint i = start; i < end; i += blockDim.x) {
    float tmp = input[i] - s_mean;
    val += tmp * tmp;
  }
  __shared__ float s_var;
  reduce_res = blockReduceSum(val);
  if (threadIdx.x == 0)
    s_var = rsqrtf(reduce_res / float(hidden_size) + epsilon);
  __syncthreads();

  // step 2. layer norm
  for (uint i = start; i < end; i += blockDim.x) {
    val = input[i] - s_mean;
    output[i] = val * s_var * __ldg(&scale[i - block_start]) +
                __ldg(&bias[i - block_start]);
    if (is_post_ln) {
      input[i] = output[i];
    }
  }
}

template <>
__global__ void ker_norm_layer_prepost<__half>(__half* input, __half* output,
                                               const __half* scale,
                                               const __half* bias,
                                               const int half_hidden_size,
                                               bool is_post_ln) {
  uint block_start = blockIdx.x * half_hidden_size;
  uint start = block_start + threadIdx.x;
  uint end = blockIdx.x * half_hidden_size + half_hidden_size;
  half2* pinput = (half2*)input;
  half2* poutput = (half2*)output;
  const half2* pscale = (const half2*)scale;
  const half2* pbias = (const half2*)bias;
  float mean_dim = float(half_hidden_size) * 2.f;

  float val = 0.0;
  // step 0. compute mean
  for (uint i = start; i < end; i += blockDim.x) {
    float2 local_f2 = safe_half2_to_float2(pinput[i]);
    val += local_f2.x + local_f2.y;
  }
  __shared__ float s_mean;
  float reduce_res = blockReduceSum<float>(val);
  if (threadIdx.x == 0) s_mean = reduce_res / mean_dim;
  __syncthreads();

  // step 1. compute variance
  val = 0.0;
  for (uint i = start; i < end; i += blockDim.x) {
    float2 local_f2 = safe_half2_to_float2(pinput[i]);
    float tmpx = local_f2.x - s_mean;
    float tmpy = local_f2.y - s_mean;
    val += tmpx * tmpx + tmpy * tmpy;
  }
  __shared__ float s_var;
  reduce_res = blockReduceSum(val);
  if (threadIdx.x == 0) s_var = rsqrtf(reduce_res / mean_dim + epsilon);
  __syncthreads();

  // step 2. layer norm
  for (uint i = start; i < end; i += blockDim.x) {
    float2 scale_val = __half22float2(__ldg(&pscale[i - block_start]));
    float2 bias_val = __half22float2(__ldg(&pbias[i - block_start]));
    float2 local_f2 = safe_half2_to_float2(pinput[i]);
    local_f2.x = (local_f2.x - s_mean) * s_var * scale_val.x + bias_val.x;
    local_f2.y = (local_f2.y - s_mean) * s_var * scale_val.y + bias_val.y;
    poutput[i] = __float22half2_rn(local_f2);
    if (is_post_ln) {
      pinput[i] = poutput[i];
    }
  }
}

template <typename T>
void ker_norm_layer_prepost_launcher(int token_num, int hidden_size,
                                     cudaStream_t stream, T* input, T* output,
                                     const T* scale, const T* bias,
                                     const int max_thread_per_block,
                                     bool is_post_ln) {
  ker_norm_layer_prepost<T><<<token_num, max_thread_per_block, 0, stream>>>(
      input, output, scale, bias, hidden_size, is_post_ln);
}

template <>
void ker_norm_layer_prepost_launcher<__half>(
    int token_num, int hidden_size, cudaStream_t stream, __half* input,
    __half* output, const __half* scale, const __half* bias,
    const int max_thread_per_block, bool is_post_ln) {
  ker_norm_layer_prepost<__half>
      <<<token_num, max_thread_per_block, 0, stream>>>(
          input, output, scale, bias, hidden_size / 2, is_post_ln);
}

template void ker_norm_layer_prepost_launcher<float>(
    int token_num, int hidden_size, cudaStream_t stream, float* input,
    float* output, const float* scale, const float* bias,
    const int max_thread_per_block, bool is_post_ln);

template void ker_norm_layer_prepost_launcher<__half>(
    int token_num, int hidden_size, cudaStream_t stream, __half* input,
    __half* output, const __half* scale, const __half* bias,
    const int max_thread_per_block, bool is_post_ln);

/**
@brief: ker_softmax_topk_router
softmax of gate output and route each token to topk experts
Currently, support topk = 1, 2

@thread
gridDim.x = batch_token_num
blockDim.x = first multiple of WARP_SIZE greater than expert_num

@param
gate_out: [batch_token_num, expert_num]
score_routed: [expert_num, max_token_num]
  score if the token is routed to the expert, else -1.0
expert_routed: [max_token_num * topk]
  ids of two routed experts.
*/
template <typename T>
__global__ void ker_softmax_topk_router(const T* gate_out, float* score_routed,
                                        int* expert_routed, int batch_token_num,
                                        int expert_num, int max_token_num,
                                        int topk) {
  int token_id = blockIdx.x, expert_id = threadIdx.x;
  // softmax
  float val = expert_id < expert_num
                  ? (float)__ldg(&gate_out[token_id * expert_num + expert_id])
                  : CUDA_FLOAT_INF_NEG;
  float max_val = blockReduceMax(val);
  __shared__ float smax;
  if (threadIdx.x == 0) smax = max_val;
  __syncthreads();

  float score = expert_id < expert_num ? expf(val - smax) : 0.f;
  float rsum = blockReduceSum(score);
  __shared__ float ssum;
  if (threadIdx.x == 0) ssum = rsum;
  __syncthreads();
  score /= ssum;

  // routing
  int idx = expert_id * max_token_num + token_id;
  score_routed[idx] = -1.0f;
  __shared__ int first_expert;
  __shared__ float first_score;
  if (val == smax) {
    first_expert = expert_id;
  }
  __syncthreads();
  if (expert_id == first_expert) {
    first_score = score;
    expert_routed[token_id] = expert_id;
    val = CUDA_FLOAT_INF_NEG;
  }

  if (topk == 1) {
    if (expert_id == first_expert) {
      score_routed[idx] = first_score;
    }
    return;
  }

  max_val = blockReduceMax(val);
  if (threadIdx.x == 0) smax = max_val;
  __syncthreads();
  __shared__ int second_expert;
  if (val == smax) {
    second_expert = expert_id;
  }
  __syncthreads();
  if (expert_id == second_expert) {
    expert_routed[token_id + max_token_num] = expert_id;
    score_routed[first_expert * max_token_num + token_id] =
        first_score / (first_score + score);
    score_routed[idx] = score / (first_score + score);
  }
}

template <typename T>
void ker_softmax_topk_router_launcher(int batch_token_num, int expert_num,
                                      int max_token_num, int topk,
                                      cudaStream_t stream, const T* gate_out,
                                      float* score_routed, int* expert_routed) {
  int block_dim = (expert_num + 31) >> 5 << 5;
  ker_softmax_topk_router<T><<<batch_token_num, block_dim, 0, stream>>>(
      gate_out, score_routed, expert_routed, batch_token_num, expert_num,
      max_token_num, topk);
}

template void ker_softmax_topk_router_launcher<float>(
    int batch_token_num, int expert_num, int max_token_num, int topk,
    cudaStream_t stream, const float* gate_out, float* score_routed,
    int* expert_routed);
template void ker_softmax_topk_router_launcher<__half>(
    int batch_token_num, int expert_num, int max_token_num, int topk,
    cudaStream_t stream, const __half* gate_out, float* score_routed,
    int* expert_routed);

/**
@brief: ker_reorder_tokens
reorder tokens by expert routing

@thread
gridDim.x = expert_num
gridDim.y = batch_token_num
blockDim.x = max_thread_per_block

@param
input: [batch_token_num, hidden_size]
score: [expert_num, max_token_num]
output: [expert_num, max_token_num, hidden_size]
*/
template <typename T>
__global__ void ker_reorder_tokens(const T* input, const float* score,
                                   T* output, int max_token_num,
                                   int hidden_size) {
  int expert_id = blockIdx.x, token_id = blockIdx.y;
  int score_pos = expert_id * max_token_num + token_id;
  if (__ldg(&score[score_pos]) > 0.) {
    for (std::size_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
      output[score_pos * hidden_size + i] =
          __ldg(&input[token_id * hidden_size + i]);
    }
  }
}

template <typename T>
void ker_reorder_tokens_launcher(int batch_token_num, int expert_num,
                                 int max_token_num, int hidden_size,
                                 int max_thread_per_block, cudaStream_t stream,
                                 const T* input, const float* score,
                                 T* output) {
  ker_reorder_tokens<T>
      <<<dim3(expert_num, batch_token_num), max_thread_per_block, 0, stream>>>(
          input, score, output, max_token_num, hidden_size);
}

template void ker_reorder_tokens_launcher<float>(
    int batch_token_num, int expert_num, int max_token_num, int hidden_size,
    int max_thread_per_block, cudaStream_t stream, const float* input,
    const float* score, float* output);

template void ker_reorder_tokens_launcher<__half>(
    int batch_token_num, int expert_num, int max_token_num, int hidden_size,
    int max_thread_per_block, cudaStream_t stream, const __half* input,
    const float* score, __half* output);

/**
@brief: ker_strided_bias_gelu
activated by gelu, add bias, each expert has unique bias

@thread
gridDim.x = expert_num
gridDim.y = batch_token_num
blockDim.x = max_thread_per_block

@param
input: [expert_num, max_token_num, feature_dim]
bias: [expert_num, feature_dim]
feature_dim: the dim of input feature
*/
template <typename T>
__global__ void ker_strided_bias_gelu(T* input, const T* bias, int feature_dim,
                                      int max_token_num) {
  int offset = (blockIdx.x * max_token_num + blockIdx.y) * feature_dim;
  for (int idx = threadIdx.x; idx < feature_dim; idx += blockDim.x) {
    int cur_offset = offset + idx;
    input[cur_offset] = gelu<float>(
        input[cur_offset] + __ldg(&bias[blockIdx.x * feature_dim + idx]));
  }
}

/* fp16 version */
template <>
__global__ void ker_strided_bias_gelu<__half>(__half* input, const __half* bias,
                                              int feature_dim,
                                              int max_token_num) {
  int offset = (blockIdx.x * max_token_num + blockIdx.y) * feature_dim;
  half2* pinput = (half2*)input;
  const half2* pbias = (const half2*)bias;
  for (int idx = threadIdx.x; idx < feature_dim; idx += blockDim.x) {
    int cur_offset = offset + idx;
    pinput[cur_offset] = gelu<half2>(__hadd2(
        pinput[cur_offset], __ldg(&pbias[blockIdx.x * feature_dim + idx])));
  }
}

template <typename T>
void ker_strided_bias_gelu_launcher(int batch_token_num, int expert_num,
                                    int max_token_num, int feature_dim,
                                    int block_dim, cudaStream_t stream,
                                    T* input, const T* bias) {
  ker_strided_bias_gelu<T>
      <<<dim3(expert_num, batch_token_num), block_dim, 0, stream>>>(
          input, bias, feature_dim, max_token_num);
}

template <>
void ker_strided_bias_gelu_launcher<__half>(int batch_token_num, int expert_num,
                                            int max_token_num, int feature_dim,
                                            int block_dim, cudaStream_t stream,
                                            __half* input, const __half* bias) {
  ker_strided_bias_gelu<__half>
      <<<dim3(expert_num, batch_token_num), block_dim, 0, stream>>>(
          input, bias, feature_dim / 2, max_token_num);
}

template void ker_strided_bias_gelu_launcher<float>(
    int batch_token_num, int expert_num, int max_token_num, int feature_dim,
    int block_dim, cudaStream_t stream, float* input, const float* bias);

template void ker_strided_bias_gelu_launcher<__half>(
    int batch_token_num, int expert_num, int max_token_num, int feature_dim,
    int block_dim, cudaStream_t stream, __half* input, const __half* bias);

/**
@brief: ker_strided_bias_relu
activated by relu, add bias, each expert has unique bias

@thread
gridDim.x = expert_num
gridDim.y = batch_token_num
blockDim.x = max_thread_per_block

@param
input: [expert_num, max_token_num, feature_dim]
bias: [expert_num, feature_dim]
feature_dim: the dim of input feature
*/
template <typename T>
__global__ void ker_strided_bias_relu(T* input, const T* bias, int feature_dim,
                                      int max_token_num) {
  int offset = (blockIdx.x * max_token_num + blockIdx.y) * feature_dim;
  for (int idx = threadIdx.x; idx < feature_dim; idx += blockDim.x) {
    int cur_offset = offset + idx;
    input[cur_offset] =
        max(input[cur_offset] + __ldg(&bias[blockIdx.x * feature_dim + idx]),
            (T)0.f);
  }
}

template <>
__global__ void ker_strided_bias_relu<__half>(__half* input, const __half* bias,
                                              int feature_dim,
                                              int max_token_num) {
  int offset = (blockIdx.x * max_token_num + blockIdx.y) * feature_dim;
  half2* pinput = (half2*)input;
  const half2* pbias = (const half2*)bias;
  for (int idx = threadIdx.x; idx < feature_dim; idx += blockDim.x) {
    int cur_offset = offset + idx;
    float2 f2_inp = __half22float2(pinput[cur_offset]);
    float2 f2_bias =
        __half22float2(__ldg(&pbias[blockIdx.x * feature_dim + idx]));
    f2_inp.x = fmaxf(f2_inp.x + f2_bias.x, 0.f);
    f2_inp.y = fmaxf(f2_inp.y + f2_bias.y, 0.f);
    pinput[cur_offset] = __float22half2_rn(f2_inp);
  }
}

template <typename T>
void ker_strided_bias_relu_launcher(int batch_token_num, int expert_num,
                                    int max_token_num, int feature_dim,
                                    int block_dim, cudaStream_t stream,
                                    T* input, const T* bias) {
  ker_strided_bias_relu<T>
      <<<dim3(expert_num, batch_token_num), block_dim, 0, stream>>>(
          input, bias, feature_dim, max_token_num);
}

template <>
void ker_strided_bias_relu_launcher<__half>(int batch_token_num, int expert_num,
                                            int max_token_num, int feature_dim,
                                            int block_dim, cudaStream_t stream,
                                            __half* input, const __half* bias) {
  ker_strided_bias_relu<__half>
      <<<dim3(expert_num, batch_token_num), block_dim, 0, stream>>>(
          input, bias, feature_dim / 2, max_token_num);
}

template void ker_strided_bias_relu_launcher<float>(
    int batch_token_num, int expert_num, int max_token_num, int feature_dim,
    int block_dim, cudaStream_t stream, float* input, const float* bias);

template void ker_strided_bias_relu_launcher<__half>(
    int batch_token_num, int expert_num, int max_token_num, int feature_dim,
    int block_dim, cudaStream_t stream, __half* input, const __half* bias);

/**
@brief: ker_bias_redirect_tokens
add second bias, each expert has unique bias,
redirect tokens to original positions, combine by score

@thread
gridDim.x = batch_token_num
blockDim.x = max_thread_per_block

@param
input: [expert_num, max_token_num, feature_dim]
bias: [expert_num, feature_dim]
score: [expert_num, max_token_num]
expert_routed: [max_token_num * topk]
output: [batch_token_num, feature_dim]
*/
template <typename T>
__global__ void ker_bias_redirect_residual(const T* input, const T* bias,
                                           const float* score,
                                           const int* expert_routed, T* output,
                                           int feature_dim, int max_token_num,
                                           int topk) {
  int expert_id = -1, token_id = blockIdx.x;
  float input_val, score_val, bias_val, output_val;
  for (int idx = threadIdx.x; idx < feature_dim; idx += blockDim.x) {
    output_val = 0.0;
    for (int k = 0; k < topk; ++k) {
      expert_id = __ldg(&expert_routed[k * max_token_num + token_id]);
      score_val = __ldg(&score[expert_id * max_token_num + token_id]);
      input_val = __ldg(
          &input[(expert_id * max_token_num + token_id) * feature_dim + idx]);
      bias_val = __ldg(&bias[expert_id * feature_dim + idx]);
      output_val += ((input_val + bias_val) * score_val);
    }
    output[token_id * feature_dim + idx] += output_val;
  }
}

template <>
__global__ void ker_bias_redirect_residual<__half>(
    const __half* input, const __half* bias, const float* score,
    const int* expert_routed, __half* output, int feature_dim,
    int max_token_num, int topk) {
  int expert_id = -1, token_id = blockIdx.x;
  const half2 *pinput = (const half2*)input, *pbias = (const half2*)bias;
  half2* poutput = (half2*)output;
  float2 f2_input_val, f2_bias_val, f2_output_val;
  float score_val;
  for (int idx = threadIdx.x; idx < feature_dim; idx += blockDim.x) {
    f2_output_val.x = 0.f;
    f2_output_val.y = 0.f;
    for (int k = 0; k < topk; ++k) {
      expert_id = __ldg(&expert_routed[k * max_token_num + token_id]);
      score_val = __ldg(&score[expert_id * max_token_num + token_id]);
      f2_input_val = __half22float2(__ldg(
          &pinput[(expert_id * max_token_num + token_id) * feature_dim + idx]));
      f2_bias_val =
          __half22float2(__ldg(&pbias[expert_id * feature_dim + idx]));
      f2_output_val.x += ((f2_input_val.x + f2_bias_val.x) * score_val);
      f2_output_val.y += ((f2_input_val.y + f2_bias_val.y) * score_val);
    }
    poutput[token_id * feature_dim + idx] =
        __hadd2(poutput[token_id * feature_dim + idx],
                __float22half2_rn(f2_output_val));
  }
}

template <typename T>
void ker_bias_redirect_residual_launcher(int hidden_size, int max_token_num,
                                         int topk, int batch_token_num,
                                         int block_dim, cudaStream_t stream,
                                         const T* input, const T* bias,
                                         const float* score,
                                         const int* expert_routed, T* output) {
  ker_bias_redirect_residual<T><<<batch_token_num, block_dim, 0, stream>>>(
      input, bias, score, expert_routed, output, hidden_size, max_token_num,
      topk);
}

template <>
void ker_bias_redirect_residual_launcher<__half>(
    int hidden_size, int max_token_num, int topk, int batch_token_num,
    int block_dim, cudaStream_t stream, const __half* input, const __half* bias,
    const float* score, const int* expert_routed, __half* output) {
  ker_bias_redirect_residual<__half><<<batch_token_num, block_dim, 0, stream>>>(
      input, bias, score, expert_routed, output, hidden_size / 2, max_token_num,
      topk);
}

template void ker_bias_redirect_residual_launcher<float>(
    int hidden_size, int max_token_num, int topk, int batch_token_num,
    int block_dim, cudaStream_t stream, const float* input, const float* bias,
    const float* score, const int* expert_routed, float* output);

template void ker_bias_redirect_residual_launcher<__half>(
    int hidden_size, int max_token_num, int topk, int batch_token_num,
    int block_dim, cudaStream_t stream, const __half* input, const __half* bias,
    const float* score, const int* expert_routed, __half* output);

}  // namespace cuda
}  // namespace lightseq
