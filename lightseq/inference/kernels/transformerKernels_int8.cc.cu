#include "transformerKernels_int8.h"

#include "common.h"
#include "transformerKernels.h"

/**
@file
Implemented the cuda kernel function and its launcher
that required by transformer model.
Currently, fp16 and fp32 versions are provided
*/
namespace lightseq {
namespace cuda {
__forceinline__ __host__ __device__ int8_t float2int8(float x,
                                                      float scale_div_clip_max,
                                                      float clip_max) {
  x = x > clip_max ? clip_max : (x < -clip_max ? -clip_max : x);
  return int8_t(x * scale_div_clip_max);
}

__forceinline__ __host__ __device__ int8_t posfloat2int8(float x, float scale,
                                                         float clip_max) {
  x = x > clip_max ? clip_max : (x < -clip_max ? -clip_max : x);
  return int8_t(x * 2 * scale / clip_max - scale);
}

template <typename T>
__global__ void quantize_tensor_kernel(const T *input, int8_t *output,
                                       int total_count, float scale,
                                       float clip_max);

template <>
__global__ void quantize_tensor_kernel<float>(const float *input,
                                              int8_t *output, int total_count,
                                              float scale_div_clip_max,
                                              float clip_max) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i * 4 >= total_count) return;

  const float4 *input4 = reinterpret_cast<const float4 *>(input);
  int32_t *output4 = reinterpret_cast<int32_t *>(output);
  float4 inp4 = input4[i];
  int32_t out4;
  int8_t *out1 = reinterpret_cast<int8_t *>(&out4);
  out1[0] = float2int8(inp4.x, scale_div_clip_max, clip_max);
  out1[1] = float2int8(inp4.y, scale_div_clip_max, clip_max);
  out1[2] = float2int8(inp4.z, scale_div_clip_max, clip_max);
  out1[3] = float2int8(inp4.w, scale_div_clip_max, clip_max);
  output4[i] = out4;
}

template <>
__global__ void quantize_tensor_kernel<__half>(const __half *input,
                                               int8_t *output, int total_count,
                                               float scale_div_clip_max,
                                               float clip_max) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i * 8 >= total_count) return;

  const float4 *input4 = reinterpret_cast<const float4 *>(input);
  int64_t *output4 = reinterpret_cast<int64_t *>(output);
  float4 inp4 = input4[i];
  int64_t out8;
  __half *inp_h = reinterpret_cast<__half *>(&inp4);
  int8_t *out1 = reinterpret_cast<int8_t *>(&out8);
#pragma unroll
  for (uint j = 0; j < 8; ++j) {
    out1[j] = float2int8(__half2float(inp_h[j]), scale_div_clip_max, clip_max);
  }
  output4[i] = out8;
}

template <>
void launch_quantize_tensor<float>(const float *input, int8_t *output,
                                   int total_count, float scale, float clip_max,
                                   cudaStream_t &stream) {
  int grid_dim = total_count >> 12;
  quantize_tensor_kernel<<<grid_dim + 1, 1024, 0, stream>>>(
      input, output, total_count, scale / clip_max, clip_max);
}

template <>
void launch_quantize_tensor<__half>(const __half *input, int8_t *output,
                                    int total_count, float scale,
                                    float clip_max, cudaStream_t &stream) {
  int grid_dim = total_count >> 13;
  quantize_tensor_kernel<<<grid_dim + 1, 1024, 0, stream>>>(
      input, output, total_count, scale / clip_max, clip_max);
}

template <typename T>
__global__ void dequantize_tensor_kernel(const int32_t *input, T *output,
                                         int total_count,
                                         float scale_div_clip_max,
                                         float clip_max);

template <>
__global__ void dequantize_tensor_kernel<float>(const int32_t *input,
                                                float *output, int total_count,
                                                float scale_div_clip_max,
                                                float clip_max) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i * 4 >= total_count) return;

  const int4 *input4 = reinterpret_cast<const int4 *>(input);
  float4 *output4 = reinterpret_cast<float4 *>(output);
  int4 inp4 = input4[i];
  float4 out4;
  out4.x = float(inp4.x) / scale_div_clip_max;
  out4.y = float(inp4.y) / scale_div_clip_max;
  out4.z = float(inp4.z) / scale_div_clip_max;
  out4.w = float(inp4.w) / scale_div_clip_max;
  output4[i] = out4;
}

template <>
__global__ void dequantize_tensor_kernel<__half>(const int32_t *input,
                                                 __half *output,
                                                 int total_count,
                                                 float scale_div_clip_max,
                                                 float clip_max) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i * 8 >= total_count) return;

  const long4 *input4 = reinterpret_cast<const long4 *>(input);
  float4 *output4 = reinterpret_cast<float4 *>(output);
  long4 inp4 = input4[i];
  float4 out4;
  int32_t *inp1 = reinterpret_cast<int32_t *>(&inp4);
  __half *out1 = reinterpret_cast<__half *>(&out4);
#pragma unroll
  for (uint j = 0; j < 8; ++j) {
    out1[j] = __float2half(float(inp1[j]) / scale_div_clip_max);
  }
  output4[i] = out4;
}

template <>
void launch_dequantize_tensor<float>(const int32_t *input, float *output,
                                     int total_count, float scale,
                                     float clip_max, cudaStream_t &stream) {
  int grid_dim = total_count >> 12;
  dequantize_tensor_kernel<<<grid_dim + 1, 1024, 0, stream>>>(
      input, output, total_count, scale / clip_max, clip_max);
}

template <>
void launch_dequantize_tensor<__half>(const int32_t *input, __half *output,
                                      int total_count, float scale,
                                      float clip_max, cudaStream_t &stream) {
  int grid_dim = total_count >> 13;
  dequantize_tensor_kernel<<<grid_dim + 1, 1024, 0, stream>>>(
      input, output, total_count, scale / clip_max, clip_max);
}

template <typename T>
__global__ void ker_norm_layer_resual_int8O(
    T *input, int8_t *output, const T *scale, const T *bias,
    const T *residual_bias, const int hidden_size, float scale_div_clip_max,
    float clip_max, bool is_post_ln, bool output_col32) {
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

  float output_f;

  // step 2. layer norm
  for (uint i = start; i < end; i += blockDim.x) {
    val = input[i] - s_mean;
    output_f = val * s_var * __ldg(&scale[i - block_start]) +
               __ldg(&bias[i - block_start]);
    int8_t res = float2int8(output_f, scale_div_clip_max, clip_max);
    if (output_col32) {
      int row_id = blockIdx.x;
      int col_id = i - start;
      int col32_index =
          row_major2flat_col32(row_id, col_id, blockDim.x, hidden_size);
      output[col32_index] = res;
    } else {
      output[i] = res;
    }
    if (is_post_ln) {
      input[i] = output_f + __ldg(&residual_bias[i - block_start]);
    } else {
      input[i] += __ldg(&residual_bias[i - block_start]);
    }
  }
}

template <>
__global__ void ker_norm_layer_resual_int8O<__half>(
    __half *input, int8_t *output, const __half *scale, const __half *bias,
    const __half *residual_bias, const int half_hidden_size,
    float scale_div_clip_max, float clip_max, bool is_post_ln,
    bool output_col32) {
  uint block_start = blockIdx.x * half_hidden_size;
  uint start = block_start + threadIdx.x;
  uint end = blockIdx.x * half_hidden_size + half_hidden_size;
  half2 *pinput = (half2 *)input;
  char2 *poutput = (char2 *)output;
  const half2 *pscale = (const half2 *)scale;
  const half2 *pbias = (const half2 *)bias;
  const half2 *presidual_bias = (const half2 *)residual_bias;
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

  char2 output_c2;

  // step 2. layer norm
  for (uint i = start; i < end; i += blockDim.x) {
    float2 scale_val = __half22float2(__ldg(&pscale[i - block_start]));
    float2 bias_val = __half22float2(__ldg(&pbias[i - block_start]));
    float2 local_f2 = safe_half2_to_float2(pinput[i]);
    local_f2.x = (local_f2.x - s_mean) * s_var * scale_val.x + bias_val.x;
    local_f2.y = (local_f2.y - s_mean) * s_var * scale_val.y + bias_val.y;
    output_c2.x = float2int8(local_f2.x, scale_div_clip_max, clip_max);
    output_c2.y = float2int8(local_f2.y, scale_div_clip_max, clip_max);

    if (output_col32) {
      int row_id = blockIdx.x;
      int col_id = (i - start) * 2;
      int col32_index = row_major2flat_col32(row_id, col_id, blockDim.x,
                                             half_hidden_size * 2);
      poutput[col32_index >> 1] = output_c2;
    } else {
      poutput[i] = output_c2;
    }

    if (!is_post_ln) {
      local_f2 = safe_half2_to_float2(pinput[i]);
    }
    float2 residual_bias_val =
        __half22float2(__ldg(&presidual_bias[i - block_start]));
    float2 new_input_f2;
    new_input_f2.x = local_f2.x + residual_bias_val.x;
    new_input_f2.y = local_f2.y + residual_bias_val.y;
    pinput[i] = __float22half2_rn(new_input_f2);
  }
}

template <typename T>
void ker_norm_layer_resual_int8O_launcher(int token_num, int hidden_size,
                                          cudaStream_t stream, T *input,
                                          int8_t *output, const T *scale,
                                          const T *bias, const T *residual_bias,
                                          const int max_thread_per_block,
                                          float quant_scale, float clip_max,
                                          bool is_post_ln, bool output_col32) {
  ker_norm_layer_resual_int8O<T>
      <<<token_num, max_thread_per_block, 0, stream>>>(
          input, output, scale, bias, residual_bias, hidden_size,
          quant_scale / clip_max, clip_max, is_post_ln, output_col32);
}

template <>
void ker_norm_layer_resual_int8O_launcher<__half>(
    int token_num, int hidden_size, cudaStream_t stream, __half *input,
    int8_t *output, const __half *scale, const __half *bias,
    const __half *residual_bias, const int max_thread_per_block,
    float quant_scale, float clip_max, bool is_post_ln, bool output_col32) {
  ker_norm_layer_resual_int8O<__half>
      <<<token_num, max_thread_per_block, 0, stream>>>(
          input, output, scale, bias, residual_bias, hidden_size / 2,
          quant_scale / clip_max, clip_max, is_post_ln, output_col32);
}

template void ker_norm_layer_resual_int8O_launcher<float>(
    int token_num, int hidden_size, cudaStream_t stream, float *input,
    int8_t *output, const float *scale, const float *bias,
    const float *residual_bias, const int max_thread_per_block,
    float quant_scale, float clip_max, bool is_post_ln, bool output_col32);

template void ker_norm_layer_resual_int8O_launcher<__half>(
    int token_num, int hidden_size, cudaStream_t stream, __half *input,
    int8_t *output, const __half *scale, const __half *bias,
    const __half *residual_bias, const int max_thread_per_block,
    float quant_scale, float clip_max, bool is_post_ln, bool output_col32);

template <typename T>
__global__ void ker_bias_gelu_int32I_int8O(int32_t *input, int8_t *output,
                                           const T *bias, int total_count,
                                           int feature_dim,
                                           float in_scale_div_clip_max,
                                           float out_scale_div_clip_max,
                                           float out_clip_max) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 4 >= total_count) return;

  char4 *out4 = reinterpret_cast<char4 *>(output);
  const int4 *data4 = reinterpret_cast<const int4 *>(input);
  const float4 *bias4 = reinterpret_cast<const float4 *>(bias);
  int bias_i = i % (feature_dim >> 2);

  const int4 input4 = data4[i];
  const float4 b4 = __ldg(&bias4[bias_i]);
  float4 output4;

  output4.x = gelu<float>(float(input4.x) / in_scale_div_clip_max + b4.x);
  output4.y = gelu<float>(float(input4.y) / in_scale_div_clip_max + b4.y);
  output4.z = gelu<float>(float(input4.z) / in_scale_div_clip_max + b4.z);
  output4.w = gelu<float>(float(input4.w) / in_scale_div_clip_max + b4.w);

  char4 out_i4;
  out_i4.x = float2int8(output4.x, out_scale_div_clip_max, out_clip_max);
  out_i4.y = float2int8(output4.y, out_scale_div_clip_max, out_clip_max);
  out_i4.z = float2int8(output4.z, out_scale_div_clip_max, out_clip_max);
  out_i4.w = float2int8(output4.w, out_scale_div_clip_max, out_clip_max);
  out4[i] = out_i4;
}

/* fp16 version */
template <>
__global__ void ker_bias_gelu_int32I_int8O<__half>(
    int32_t *input, int8_t *output, const __half *bias, int total_count,
    int feature_dim, float in_scale_div_clip_max, float out_scale_div_clip_max,
    float out_clip_max) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 8 >= total_count) return;

  const long4 *vals_long4 = reinterpret_cast<const long4 *>(input);
  int64_t *outs_i8 = reinterpret_cast<int64_t *>(output);
  const float4 *bias4 = reinterpret_cast<const float4 *>(bias);

  int bias_i = i % (feature_dim >> 3);
  long4 val_long4 = vals_long4[i];
  int32_t *val1 = reinterpret_cast<int32_t *>(&val_long4);
  const float4 b4 = __ldg(&bias4[bias_i]);
  const __half *b_half = reinterpret_cast<const __half *>(&b4);
  int64_t out_i8;
  int8_t *out_i1 = reinterpret_cast<int8_t *>(&out_i8);

#pragma unroll
  for (uint j = 0; j < 8; ++j) {
    float out_f;
    out_f = gelu<float>(float(val1[j]) / in_scale_div_clip_max +
                        __half2float(b_half[j]));
    out_i1[j] = float2int8(out_f, out_scale_div_clip_max, out_clip_max);
  }
  outs_i8[i] = out_i8;
}

template <typename T>
void ker_bias_gelu_int32I_int8O_launcher(int batch_token_num,
                                         cudaStream_t stream, int32_t *input,
                                         int8_t *output, const T *bias,
                                         int feature_dim, float in_scale,
                                         float in_clip_max, float out_scale,
                                         float out_clip_max) {
  int total_count = batch_token_num * feature_dim;
  int grid_dim = total_count >> 10;
  ker_bias_gelu_int32I_int8O<T><<<grid_dim + 1, 256, 0, stream>>>(
      input, output, bias, total_count, feature_dim, in_scale / in_clip_max,
      out_scale / out_clip_max, out_clip_max);
}

template <>
void ker_bias_gelu_int32I_int8O_launcher<__half>(
    int batch_token_num, cudaStream_t stream, int32_t *input, int8_t *output,
    const __half *bias, int feature_dim, float in_scale, float in_clip_max,
    float out_scale, float out_clip_max) {
  int total_count = batch_token_num * feature_dim;
  int grid_dim = total_count >> 11;
  ker_bias_gelu_int32I_int8O<__half><<<grid_dim + 1, 256, 0, stream>>>(
      input, output, bias, total_count, feature_dim, in_scale / in_clip_max,
      out_scale / out_clip_max, out_clip_max);
}

template void ker_bias_gelu_int32I_int8O_launcher<float>(
    int batch_token_num, cudaStream_t stream, int32_t *input, int8_t *output,
    const float *bias, int feature_dim, float in_scale, float in_clip_max,
    float out_scale, float out_clip_max);

template void ker_bias_gelu_int32I_int8O_launcher<__half>(
    int batch_token_num, cudaStream_t stream, int32_t *input, int8_t *output,
    const __half *bias, int feature_dim, float in_scale, float in_clip_max,
    float out_scale, float out_clip_max);

template <typename T>
__global__ void ker_bias_relu_int32I_int8O(int32_t *input, int8_t *output,
                                           const T *bias, int total_count,
                                           int feature_dim,
                                           float in_scale_div_clip_max,
                                           float out_scale_div_clip_max,
                                           float out_clip_max) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 4 >= total_count) return;

  char4 *out4 = reinterpret_cast<char4 *>(output);
  const int4 *data4 = reinterpret_cast<const int4 *>(input);
  const float4 *bias4 = reinterpret_cast<const float4 *>(bias);
  int bias_i = i % (feature_dim >> 2);

  const int4 input4 = data4[i];
  const float4 b4 = __ldg(&bias4[bias_i]);
  float4 output4;

  output4.x = max(float(input4.x) / in_scale_div_clip_max + b4.x, (T)0.f);
  output4.y = max(float(input4.y) / in_scale_div_clip_max + b4.y, (T)0.f);
  output4.z = max(float(input4.z) / in_scale_div_clip_max + b4.z, (T)0.f);
  output4.w = max(float(input4.w) / in_scale_div_clip_max + b4.w, (T)0.f);

  char4 out_i4;
  out_i4.x = float2int8(output4.x, out_scale_div_clip_max, out_clip_max);
  out_i4.y = float2int8(output4.y, out_scale_div_clip_max, out_clip_max);
  out_i4.z = float2int8(output4.z, out_scale_div_clip_max, out_clip_max);
  out_i4.w = float2int8(output4.w, out_scale_div_clip_max, out_clip_max);
  out4[i] = out_i4;
}

/* fp16 version */
template <>
__global__ void ker_bias_relu_int32I_int8O<__half>(
    int32_t *input, int8_t *output, const __half *bias, int total_count,
    int feature_dim, float in_scale_div_clip_max, float out_scale_div_clip_max,
    float out_clip_max) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 8 >= total_count) return;

  const long4 *vals_long4 = reinterpret_cast<const long4 *>(input);
  int64_t *outs_i8 = reinterpret_cast<int64_t *>(output);
  const float4 *bias4 = reinterpret_cast<const float4 *>(bias);

  int bias_i = i % (feature_dim >> 3);
  long4 val_long4 = vals_long4[i];
  int32_t *val1 = reinterpret_cast<int32_t *>(&val_long4);
  const float4 b4 = __ldg(&bias4[bias_i]);
  const __half *b_half = reinterpret_cast<const __half *>(&b4);
  int64_t out_i8;
  int8_t *out_i1 = reinterpret_cast<int8_t *>(&out_i8);

#pragma unroll
  for (uint j = 0; j < 8; ++j) {
    float out_f;
    out_f =
        max(float(val1[j]) / in_scale_div_clip_max + __half2float(b_half[j]),
            (float)0.f);
    out_i1[j] = float2int8(out_f, out_scale_div_clip_max, out_clip_max);
  }
  outs_i8[i] = out_i8;
}

template <typename T>
void ker_bias_relu_int32I_int8O_launcher(int batch_token_num,
                                         cudaStream_t stream, int32_t *input,
                                         int8_t *output, const T *bias,
                                         int feature_dim, float in_scale,
                                         float in_clip_max, float out_scale,
                                         float out_clip_max) {
  int total_count = batch_token_num * feature_dim;
  int grid_dim = total_count >> 10;
  ker_bias_relu_int32I_int8O<T><<<grid_dim + 1, 256, 0, stream>>>(
      input, output, bias, total_count, feature_dim, in_scale / in_clip_max,
      out_scale / out_clip_max, out_clip_max);
}

template <>
void ker_bias_relu_int32I_int8O_launcher<__half>(
    int batch_token_num, cudaStream_t stream, int32_t *input, int8_t *output,
    const __half *bias, int feature_dim, float in_scale, float in_clip_max,
    float out_scale, float out_clip_max) {
  int total_count = batch_token_num * feature_dim;
  int grid_dim = total_count >> 11;
  ker_bias_relu_int32I_int8O<__half><<<grid_dim + 1, 256, 0, stream>>>(
      input, output, bias, total_count, feature_dim, in_scale / in_clip_max,
      out_scale / out_clip_max, out_clip_max);
}

template void ker_bias_relu_int32I_int8O_launcher<float>(
    int batch_token_num, cudaStream_t stream, int32_t *input, int8_t *output,
    const float *bias, int feature_dim, float in_scale, float in_clip_max,
    float out_scale, float out_clip_max);

template void ker_bias_relu_int32I_int8O_launcher<__half>(
    int batch_token_num, cudaStream_t stream, int32_t *input, int8_t *output,
    const __half *bias, int feature_dim, float in_scale, float in_clip_max,
    float out_scale, float out_clip_max);

template <typename T>
__global__ void ker_residual_int32I(int32_t *input, T *output, int total_count,
                                    float scale_div_clip_max) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 4 >= total_count) return;

  float4 *out4 = reinterpret_cast<float4 *>(output);
  const int4 *data4 = reinterpret_cast<const int4 *>(input);
  const int4 input4 = data4[i];
  float4 output4 = out4[i];

  output4.x += float(input4.x) / scale_div_clip_max;
  output4.y += float(input4.y) / scale_div_clip_max;
  output4.z += float(input4.z) / scale_div_clip_max;
  output4.w += float(input4.w) / scale_div_clip_max;

  out4[i] = output4;
}

/* fp16 version */
template <>
__global__ void ker_residual_int32I<__half>(int32_t *input, __half *output,
                                            int total_count,
                                            float scale_div_clip_max) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 8 >= total_count) return;

  const long4 *vals_long4 = reinterpret_cast<const long4 *>(input);
  float4 *outs_h8 = reinterpret_cast<float4 *>(output);
  long4 val_long4 = vals_long4[i];
  int32_t *val1 = reinterpret_cast<int32_t *>(&val_long4);
  float4 out_h8 = outs_h8[i];
  __half *out_h1 = reinterpret_cast<__half *>(&out_h8);

#pragma unroll
  for (uint j = 0; j < 8; ++j) {
    out_h1[j] =
        __hadd(out_h1[j], __float2half(float(val1[j]) / scale_div_clip_max));
  }
  outs_h8[i] = out_h8;
}

template <typename T>
void ker_residual_int32I_launcher(int32_t *input, T *output, int total_ele_num,
                                  float quant_scale, float clip_max,
                                  cudaStream_t stream) {
  int grid_dim = total_ele_num >> 10;
  ker_residual_int32I<T><<<grid_dim + 1, 256, 0, stream>>>(
      input, output, total_ele_num, quant_scale / clip_max);
}

template <>
void ker_residual_int32I_launcher<__half>(int32_t *input, __half *output,
                                          int total_ele_num, float quant_scale,
                                          float clip_max, cudaStream_t stream) {
  int grid_dim = total_ele_num >> 11;
  ker_residual_int32I<__half><<<grid_dim + 1, 256, 0, stream>>>(
      input, output, total_ele_num, quant_scale / clip_max);
}

template void ker_residual_int32I_launcher<float>(int32_t *input, float *output,
                                                  int total_ele_num,
                                                  float quant_scale,
                                                  float clip_max,
                                                  cudaStream_t stream);

template void ker_residual_int32I_launcher<__half>(
    int32_t *input, __half *output, int total_ele_num, float quant_scale,
    float clip_max, cudaStream_t stream);

template <typename T>
__global__ void ker_arrange_encself_qkv_int32I(const int32_t *ori_qkv,
                                               const T *qkv_bias, T *new_qkv,
                                               int max_batch_dim,
                                               int batch_seq_len,
                                               int dim_per_head, int head_num,
                                               float scale_div_clip_max) {
  int hidden_size = dim_per_head * head_num;
  int batch_id = blockIdx.x / batch_seq_len;
  int token_id = blockIdx.x % batch_seq_len;
  int qkv_offset = max_batch_dim * blockIdx.y;
  for (std::size_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    int target_id = targetid_4dim(batch_id, head_id, token_id, dim_id, head_num,
                                  batch_seq_len, dim_per_head);
    new_qkv[qkv_offset + target_id] =
        float(
            ori_qkv[(blockIdx.x * gridDim.y + blockIdx.y) * hidden_size + i]) /
            scale_div_clip_max +
        __ldg(&qkv_bias[blockIdx.y * hidden_size + i]);
  }
}

template <>
__global__ void ker_arrange_encself_qkv_int32I<__half>(
    const int32_t *ori_qkv, const __half *qkv_bias, __half *new_qkv,
    int max_batch_dim, int batch_seq_len, int dim_per_head, int head_num,
    float scale_div_clip_max) {
  int hidden_size = dim_per_head * head_num;
  int batch_id = blockIdx.x / batch_seq_len;
  int token_id = blockIdx.x % batch_seq_len;
  for (std::size_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    int qkv_offset = max_batch_dim * blockIdx.y;
    int target_id = targetid_4dim(batch_id, head_id, token_id, dim_id, head_num,
                                  batch_seq_len, dim_per_head);

    const int2 *p_ori_qkv = (const int2 *)ori_qkv;
    const half2 *p_bias = (const half2 *)qkv_bias;
    half2 *p_new_qkv = (half2 *)new_qkv;
    int2 ori_qkv_i2 =
        p_ori_qkv[(blockIdx.x * gridDim.y + blockIdx.y) * hidden_size + i];
    half2 ori_qkv_h2;
    ori_qkv_h2.x = __float2half(float(ori_qkv_i2.x) / scale_div_clip_max);
    ori_qkv_h2.y = __float2half(float(ori_qkv_i2.y) / scale_div_clip_max);
    p_new_qkv[qkv_offset + target_id] =
        __hadd2(ori_qkv_h2, __ldg(&p_bias[blockIdx.y * hidden_size + i]));
  }
}

template <typename T>
void ker_arrange_encself_qkv_int32I_launcher(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const int32_t *ori_qkv, const T *qkv_bias, T *new_qkv, int max_batch_dim,
    int batch_seq_len, int dim_per_head, int head_num, int max_thread_per_block,
    float quant_scale, float clip_max) {
  ker_arrange_encself_qkv_int32I<T>
      <<<dim3(batch_token_num, 3), max_thread_per_block, 0, stream>>>(
          ori_qkv, qkv_bias, new_qkv, max_batch_dim, batch_seq_len,
          dim_per_head, head_num, quant_scale / clip_max);
}

template <>
void ker_arrange_encself_qkv_int32I_launcher<__half>(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const int32_t *ori_qkv, const __half *qkv_bias, __half *new_qkv,
    int max_batch_dim, int batch_seq_len, int dim_per_head, int head_num,
    int max_thread_per_block, float quant_scale, float clip_max) {
  ker_arrange_encself_qkv_int32I<__half>
      <<<dim3(batch_token_num, 3), max_thread_per_block, 0, stream>>>(
          ori_qkv, qkv_bias, new_qkv, max_batch_dim / 2, batch_seq_len,
          dim_per_head / 2, head_num, quant_scale / clip_max);
}

template void ker_arrange_encself_qkv_int32I_launcher<float>(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const int32_t *ori_qkv, const float *qkv_bias, float *new_qkv,
    int max_batch_dim, int batch_seq_len, int dim_per_head, int head_num,
    int max_thread_per_block, float quant_scale, float clip_max);

template void ker_arrange_encself_qkv_int32I_launcher<__half>(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const int32_t *ori_qkv, const __half *qkv_bias, __half *new_qkv,
    int max_batch_dim, int batch_seq_len, int dim_per_head, int head_num,
    int max_thread_per_block, float quant_scale, float clip_max);

template <typename T>
__global__ void ker_arrange_atten_output_int8O(const T *ori_q, int8_t *new_q,
                                               int beam_size, int dim_per_head,
                                               int head_num,
                                               float scale_div_clip_max,
                                               float clip_max) {
  int hidden_size = dim_per_head * head_num;
  int batch_id = blockIdx.x / beam_size;
  // note, for encoder, beam_id is token_id; for decoder, beam_id is beam_id
  int beam_id = blockIdx.x % beam_size;
  for (std::size_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    new_q[blockIdx.x * hidden_size + i] =
        float2int8(ori_q[targetid_4dim(batch_id, head_id, beam_id, dim_id,
                                       head_num, beam_size, dim_per_head)],
                   scale_div_clip_max, clip_max);
  }
}

template <>
__global__ void ker_arrange_atten_output_int8O<__half>(
    const __half *ori_q, int8_t *new_q, int beam_size, int dim_per_head,
    int head_num, float scale_div_clip_max, float clip_max) {
  int batch_id = blockIdx.x / beam_size;
  // note, for encoder, beam_id is token_id; for decoder, beam_id is beam_id
  int beam_id = blockIdx.x % beam_size;
  int half_hidden_size = dim_per_head * head_num;
  for (std::size_t i = threadIdx.x; i < half_hidden_size; i += blockDim.x) {
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    const half2 *p_ori_q = (const half2 *)ori_q;
    half2 v_ori_q;
    char2 *p_new_q = (char2 *)new_q;
    char2 v_new_q;
    v_ori_q = p_ori_q[targetid_4dim(batch_id, head_id, beam_id, dim_id,
                                    head_num, beam_size, dim_per_head)];
    v_new_q.x = float2int8(float(v_ori_q.x), scale_div_clip_max, clip_max);
    v_new_q.y = float2int8(float(v_ori_q.y), scale_div_clip_max, clip_max);
    p_new_q[blockIdx.x * half_hidden_size + i] = v_new_q;
  }
}

template <typename T>
void ker_arrange_atten_output_int8O_launcher(
    int batch_token_num, int hidden_size, cudaStream_t stream, const T *ori_q,
    int8_t *new_q, int beam_size, int dim_per_head, int head_num,
    int max_thread_per_block, float quant_scale, float clip_max) {
  ker_arrange_atten_output_int8O<T>
      <<<batch_token_num, max_thread_per_block, 0, stream>>>(
          ori_q, new_q, beam_size, dim_per_head, head_num,
          quant_scale / clip_max, clip_max);
}

template <>
void ker_arrange_atten_output_int8O_launcher<__half>(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const __half *ori_q, int8_t *new_q, int beam_size, int dim_per_head,
    int head_num, int max_thread_per_block, float quant_scale, float clip_max) {
  ker_arrange_atten_output_int8O<__half>
      <<<batch_token_num, max_thread_per_block, 0, stream>>>(
          ori_q, new_q, beam_size, dim_per_head / 2, head_num,
          quant_scale / clip_max, clip_max);
}

template void ker_arrange_atten_output_int8O_launcher<float>(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const float *ori_q, int8_t *new_q, int beam_size, int dim_per_head,
    int head_num, int max_thread_per_block, float quant_scale, float clip_max);

template void ker_arrange_atten_output_int8O_launcher<__half>(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const __half *ori_q, int8_t *new_q, int beam_size, int dim_per_head,
    int head_num, int max_thread_per_block, float quant_scale, float clip_max);

template <typename T>
__global__ void ker_arrange_decself_qkv_int32I(const int32_t *ori_qkv,
                                               const T *qkv_bias, T *new_q,
                                               T *new_k, T *new_v, int head_num,
                                               int dim_per_head, int max_step,
                                               int step_id,
                                               float scale_div_clip_max) {
  int hidden_size = dim_per_head * head_num;
  for (std::size_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    // blockdim is equal to hidden_size
    T val =
        float(
            ori_qkv[(blockIdx.x * gridDim.y + blockIdx.y) * hidden_size + i]) /
            scale_div_clip_max +
        __ldg(&qkv_bias[blockIdx.y * hidden_size + i]);
    int seq_id =
        blockIdx.x;  // obvious， seq_id = batch_id * beam_size + beam_id
    if (blockIdx.y == 0) {
      // for query
      new_q[seq_id * hidden_size + i] = val;
      return;
    }
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    int target_id = targetid_4dim(seq_id, head_id, step_id, dim_id, head_num,
                                  max_step, dim_per_head);
    if (blockIdx.y == 1) {
      // for key
      new_k[target_id] = val;
    } else {
      // for value
      new_v[target_id] = val;
    }
  }
}

template <>
__global__ void ker_arrange_decself_qkv_int32I<__half>(
    const int32_t *ori_qkv, const __half *qkv_bias, __half *new_q,
    __half *new_k, __half *new_v, int head_num, int dim_per_head, int max_step,
    int step_id, float scale_div_clip_max) {
  int half_hidden_size = dim_per_head * head_num;
  const int2 *p_qkv = (const int2 *)ori_qkv;
  const half2 *p_bias = (const half2 *)qkv_bias;
  int2 v_ori_qkv;
  half2 ori_qkv_h2;
  for (std::size_t i = threadIdx.x; i < half_hidden_size; i += blockDim.x) {
    v_ori_qkv =
        p_qkv[(blockIdx.x * gridDim.y + blockIdx.y) * half_hidden_size + i];
    ori_qkv_h2.x = __float2half(float(v_ori_qkv.x) / scale_div_clip_max);
    ori_qkv_h2.y = __float2half(float(v_ori_qkv.y) / scale_div_clip_max);
    half2 val =
        __hadd2(ori_qkv_h2, __ldg(&p_bias[blockIdx.y * half_hidden_size + i]));
    // obvious，seq_id = batch_id * beam_size + beam_id
    int seq_id = blockIdx.x;
    if (blockIdx.y == 0) {
      // for query
      ((half2 *)new_q)[seq_id * half_hidden_size + i] = val;
      return;
    }
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    int target_id = targetid_4dim(seq_id, head_id, step_id, dim_id, head_num,
                                  max_step, dim_per_head);
    if (blockIdx.y == 1) {
      // for key
      ((half2 *)new_k)[target_id] = val;
    } else {
      // for value
      ((half2 *)new_v)[target_id] = val;
    }
  }
}

template <typename T>
void ker_arrange_decself_qkv_int32I_launcher(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const int32_t *ori_qkv, const T *qkv_bias, T *new_q, T *new_k, T *new_v,
    int head_num, int dim_per_head, int max_step, int step_id,
    int max_thread_per_block, float quant_scale, float clip_max) {
  ker_arrange_decself_qkv_int32I<T>
      <<<dim3(step_token_num, 3), max_thread_per_block, 0, stream>>>(
          ori_qkv, qkv_bias, new_q, new_k, new_v, head_num, dim_per_head,
          max_step, step_id, quant_scale / clip_max);
}

template <>
void ker_arrange_decself_qkv_int32I_launcher<__half>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const int32_t *ori_qkv, const __half *qkv_bias, __half *new_q,
    __half *new_k, __half *new_v, int head_num, int dim_per_head, int max_step,
    int step_id, int max_thread_per_block, float quant_scale, float clip_max) {
  ker_arrange_decself_qkv_int32I<__half>
      <<<dim3(step_token_num, 3), max_thread_per_block, 0, stream>>>(
          ori_qkv, qkv_bias, new_q, new_k, new_v, head_num, dim_per_head / 2,
          max_step, step_id, quant_scale / clip_max);
}

template void ker_arrange_decself_qkv_int32I_launcher<float>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const int32_t *ori_qkv, const float *qkv_bias, float *new_q, float *new_k,
    float *new_v, int head_num, int dim_per_head, int max_step, int step_id,
    int max_thread_per_block, float quant_scale, float clip_max);

template void ker_arrange_decself_qkv_int32I_launcher<__half>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const int32_t *ori_qkv, const __half *qkv_bias, __half *new_q,
    __half *new_k, __half *new_v, int head_num, int dim_per_head, int max_step,
    int step_id, int max_thread_per_block, float quant_scale, float clip_max);

template <typename T>
__global__ void ker_arrange_decself_qkv_int8I(const int8_t *ori_qkv,
                                              const T *qkv_bias, T *new_q,
                                              T *new_k, T *new_v, int head_num,
                                              int dim_per_head, int max_step,
                                              int step_id,
                                              float scale_div_clip_max) {
  int hidden_size = dim_per_head * head_num;
  for (std::size_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    // blockdim is equal to hidden_size
    int row_id = blockIdx.x;
    int col_id = blockIdx.y * hidden_size + i;
    int col32_index = row_major2flat_col32(row_id, col_id, blockDim.x,
                                           blockDim.y * hidden_size);
    T val = float(ori_qkv[col32_index]) / scale_div_clip_max +
            __ldg(&qkv_bias[blockIdx.y * hidden_size + i]);
    int seq_id =
        blockIdx.x;  // obvious， seq_id = batch_id * beam_size + beam_id
    if (blockIdx.y == 0) {
      // for query
      new_q[seq_id * hidden_size + i] = val;
      return;
    }
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    int target_id = targetid_4dim(seq_id, head_id, step_id, dim_id, head_num,
                                  max_step, dim_per_head);
    if (blockIdx.y == 1) {
      // for key
      new_k[target_id] = val;
    } else {
      // for value
      new_v[target_id] = val;
    }
  }
}

template <>
__global__ void ker_arrange_decself_qkv_int8I<__half>(
    const int8_t *ori_qkv, const __half *qkv_bias, __half *new_q, __half *new_k,
    __half *new_v, int head_num, int dim_per_head, int max_step, int step_id,
    float scale_div_clip_max) {
  int half_hidden_size = dim_per_head * head_num;
  const char2 *p_qkv = reinterpret_cast<const char2 *>(ori_qkv);
  const half2 *p_bias = reinterpret_cast<const half2 *>(qkv_bias);
  char2 v_ori_qkv;
  half2 ori_qkv_h2;
  for (std::size_t i = threadIdx.x; i < half_hidden_size; i += blockDim.x) {
    int row_id = blockIdx.x;
    int col_id = (blockIdx.y * half_hidden_size + i) * 2;
    int col32_index = row_major2flat_col32(row_id, col_id, blockDim.x,
                                           blockDim.y * half_hidden_size) >>
                      1;
    v_ori_qkv = p_qkv[col32_index];
    ori_qkv_h2.x = __float2half(float(v_ori_qkv.x) / scale_div_clip_max);
    ori_qkv_h2.y = __float2half(float(v_ori_qkv.y) / scale_div_clip_max);
    half2 val =
        __hadd2(ori_qkv_h2, __ldg(&p_bias[blockIdx.y * half_hidden_size + i]));
    // obvious，seq_id = batch_id * beam_size + beam_id
    int seq_id = blockIdx.x;
    if (blockIdx.y == 0) {
      // for query
      ((half2 *)new_q)[seq_id * half_hidden_size + i] = val;
      return;
    }
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    int target_id = targetid_4dim(seq_id, head_id, step_id, dim_id, head_num,
                                  max_step, dim_per_head);
    if (blockIdx.y == 1) {
      // for key
      ((half2 *)new_k)[target_id] = val;
    } else {
      // for value
      ((half2 *)new_v)[target_id] = val;
    }
  }
}

template <typename T>
void ker_arrange_decself_qkv_int8I_launcher(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const int8_t *ori_qkv, const T *qkv_bias, T *new_q, T *new_k, T *new_v,
    int head_num, int dim_per_head, int max_step, int step_id,
    int max_thread_per_block, float quant_scale, float clip_max) {
  ker_arrange_decself_qkv_int8I<T>
      <<<dim3(step_token_num, 3), max_thread_per_block, 0, stream>>>(
          ori_qkv, qkv_bias, new_q, new_k, new_v, head_num, dim_per_head,
          max_step, step_id, quant_scale / clip_max);
}

template <>
void ker_arrange_decself_qkv_int8I_launcher<__half>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const int8_t *ori_qkv, const __half *qkv_bias, __half *new_q, __half *new_k,
    __half *new_v, int head_num, int dim_per_head, int max_step, int step_id,
    int max_thread_per_block, float quant_scale, float clip_max) {
  ker_arrange_decself_qkv_int8I<__half>
      <<<dim3(step_token_num, 3), max_thread_per_block, 0, stream>>>(
          ori_qkv, qkv_bias, new_q, new_k, new_v, head_num, dim_per_head / 2,
          max_step, step_id, quant_scale / clip_max);
}

template void ker_arrange_decself_qkv_int8I_launcher<float>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const int8_t *ori_qkv, const float *qkv_bias, float *new_q, float *new_k,
    float *new_v, int head_num, int dim_per_head, int max_step, int step_id,
    int max_thread_per_block, float quant_scale, float clip_max);

template void ker_arrange_decself_qkv_int8I_launcher<__half>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const int8_t *ori_qkv, const __half *qkv_bias, __half *new_q, __half *new_k,
    __half *new_v, int head_num, int dim_per_head, int max_step, int step_id,
    int max_thread_per_block, float quant_scale, float clip_max);

template <typename T>
__global__ void ker_arrange_encdec_q_int32I(const int32_t *ori_q,
                                            const T *q_bias, T *new_q,
                                            int beam_size, int dim_per_head,
                                            int head_num,
                                            float scale_div_clip_max) {
  int hidden_size = dim_per_head * head_num;
  for (std::size_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    T val = float(ori_q[blockIdx.x * hidden_size + i]) / scale_div_clip_max +
            __ldg(&q_bias[i]);
    int batch_id = blockIdx.x / beam_size;
    int beam_id = blockIdx.x % beam_size;
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    new_q[targetid_4dim(batch_id, head_id, beam_id, dim_id, head_num, beam_size,
                        dim_per_head)] = val;
  }
}

template <>
__global__ void ker_arrange_encdec_q_int32I<__half>(
    const int32_t *ori_q, const __half *q_bias, __half *new_q, int beam_size,
    int dim_per_head, int head_num, float scale_div_clip_max) {
  int half_hidden_size = dim_per_head * head_num;
  for (std::size_t i = threadIdx.x; i < half_hidden_size; i += blockDim.x) {
    const int2 *p_q = (const int2 *)ori_q;
    int2 p_q_i2 = p_q[blockIdx.x * half_hidden_size + i];
    half2 p_q_h2;
    p_q_h2.x = __float2half(float(p_q_i2.x) / scale_div_clip_max);
    p_q_h2.y = __float2half(float(p_q_i2.y) / scale_div_clip_max);
    const half2 *p_bias = (const half2 *)q_bias;
    half2 val = __hadd2(p_q_h2, __ldg(&p_bias[i]));
    int batch_id = blockIdx.x / beam_size;
    int beam_id = blockIdx.x % beam_size;
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    ((half2 *)new_q)[targetid_4dim(batch_id, head_id, beam_id, dim_id, head_num,
                                   beam_size, dim_per_head)] = val;
  }
}

template <typename T>
void ker_arrange_encdec_q_int32I_launcher(int step_token_num, int hidden_size,
                                          cudaStream_t stream,
                                          const int32_t *ori_q, const T *q_bias,
                                          T *new_q, int beam_size,
                                          int dim_per_head, int head_num,
                                          int max_thread_per_block,
                                          float quant_scale, float clip_max) {
  ker_arrange_encdec_q_int32I<T>
      <<<step_token_num, max_thread_per_block, 0, stream>>>(
          ori_q, q_bias, new_q, beam_size, dim_per_head, head_num,
          quant_scale / clip_max);
}

template <>
void ker_arrange_encdec_q_int32I_launcher<__half>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const int32_t *ori_q, const __half *q_bias, __half *new_q, int beam_size,
    int dim_per_head, int head_num, int max_thread_per_block, float quant_scale,
    float clip_max) {
  ker_arrange_encdec_q_int32I<__half>
      <<<step_token_num, max_thread_per_block, 0, stream>>>(
          ori_q, q_bias, new_q, beam_size, dim_per_head / 2, head_num,
          quant_scale / clip_max);
}

template void ker_arrange_encdec_q_int32I_launcher<float>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const int32_t *ori_q, const float *q_bias, float *new_q, int beam_size,
    int dim_per_head, int head_num, int max_thread_per_block, float quant_scale,
    float clip_max);

template void ker_arrange_encdec_q_int32I_launcher<__half>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const int32_t *ori_q, const __half *q_bias, __half *new_q, int beam_size,
    int dim_per_head, int head_num, int max_thread_per_block, float quant_scale,
    float clip_max);

}  // namespace cuda
}  // namespace lightseq
