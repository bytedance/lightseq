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
__forceinline__ __device__ int8_t float2int8(float x, float quant_scale) {
  float i8_f = x * quant_scale;
  int32_t i8 = floorf(i8_f + 0.5);
  i8 = i8 < -127 ? -127 : (i8 > 127 ? 127 : i8);
  return int8_t(i8);
}

__forceinline__ __device__ int8_t posfloat2int8(float x, float quant_scale,
                                                float clip_max) {
  float i8_f = x * 2 * quant_scale - quant_scale * clip_max;
  int32_t i8 = floorf(i8_f + 0.5);
  i8 = i8 < -127 ? -127 : (i8 > 127 ? 127 : i8);
  return int8_t(i8);
}

__global__ void quantize_tensor_kernel(const float *input, int8_t *output,
                                       int batch_tokens, int hidden_size,
                                       float quant_scale, bool out_col32) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= batch_tokens * hidden_size) return;
  int output_index;
  if (out_col32) {
    int row_id = i / hidden_size;
    int col_id = i % hidden_size;
    output_index =
        row_major2flat_col32(row_id, col_id, batch_tokens, hidden_size);
  } else {
    output_index = i;
  }
  output[output_index] = float2int8(input[i], quant_scale);
}

__global__ void quantize_tensor_kernel(const __half *input, int8_t *output,
                                       int batch_tokens, int hidden_size,
                                       float quant_scale, bool out_col32) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= batch_tokens * hidden_size) return;

  int output_index;
  if (out_col32) {
    int row_id = i / hidden_size;
    int col_id = i % hidden_size;
    output_index =
        row_major2flat_col32(row_id, col_id, batch_tokens, hidden_size);
  } else {
    output_index = i;
  }
  output[output_index] = float2int8(__half2float(input[i]), quant_scale);
}

template <>
void launch_quantize_tensor<float>(const float *input, int8_t *output,
                                   int batch_tokens, int hidden_size,
                                   float quant_scale, cudaStream_t &stream,
                                   bool out_col32) {
  int grid_dim = (batch_tokens * hidden_size) >> 10;
  quantize_tensor_kernel<<<grid_dim + 1, 1024, 0, stream>>>(
      input, output, batch_tokens, hidden_size, quant_scale, out_col32);
}

template <>
void launch_quantize_tensor<__half>(const __half *input, int8_t *output,
                                    int batch_tokens, int hidden_size,
                                    float quant_scale, cudaStream_t &stream,
                                    bool out_col32) {
  int grid_dim = (batch_tokens * hidden_size) >> 10;
  quantize_tensor_kernel<<<grid_dim + 1, 1024, 0, stream>>>(
      input, output, batch_tokens, hidden_size, quant_scale, out_col32);
}

template <typename T>
__global__ void dequantize_tensor_kernel(const int32_t *input, T *output,
                                         int batch_tokens, int hidden_size,
                                         float dequant_scale, bool in_col32);

template <>
__global__ void dequantize_tensor_kernel<float>(const int32_t *input,
                                                float *output, int batch_tokens,
                                                int hidden_size,
                                                float dequant_scale,
                                                bool in_col32) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= batch_tokens * hidden_size) return;
  int input_index;
  if (in_col32) {
    int row_id = i / hidden_size;
    int col_id = i % hidden_size;
    input_index =
        row_major2flat_col32(row_id, col_id, batch_tokens, hidden_size);
  } else {
    input_index = i;
  }
  output[i] = input[input_index] * dequant_scale;
}

template <>
__global__ void dequantize_tensor_kernel<__half>(
    const int32_t *input, __half *output, int batch_tokens, int hidden_size,
    float dequant_scale, bool in_col32) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= batch_tokens * hidden_size) return;

  int in_index;
  if (in_col32) {
    int row_id = i / hidden_size;
    int col_id = i % hidden_size;
    in_index = row_major2flat_col32(row_id, col_id, batch_tokens, hidden_size);
  } else {
    in_index = i;
  }
  output[i] = __float2half(float(input[in_index]) * dequant_scale);
}

template <>
void launch_dequantize_tensor<float>(const int32_t *input, float *output,
                                     int batch_tokens, int hidden_size,
                                     float dequant_scale, cudaStream_t &stream,
                                     bool in_col32) {
  int total_count = batch_tokens * hidden_size;
  int grid_dim = total_count >> 10;
  dequantize_tensor_kernel<<<grid_dim + 1, 1024, 0, stream>>>(
      input, output, batch_tokens, hidden_size, dequant_scale, in_col32);
}

template <>
void launch_dequantize_tensor<__half>(const int32_t *input, __half *output,
                                      int batch_tokens, int hidden_size,
                                      float dequant_scale, cudaStream_t &stream,
                                      bool in_col32) {
  int total_count = batch_tokens * hidden_size;
  int grid_dim = total_count >> 10;
  dequantize_tensor_kernel<<<grid_dim + 1, 1024, 0, stream>>>(
      input, output, batch_tokens, hidden_size, dequant_scale, in_col32);
}

template <typename T>
__global__ void ker_norm_layer_resual_i8O(T *input, int8_t *output,
                                          const T *scale, const T *bias,
                                          const T *residual_bias,
                                          const int hidden_size,
                                          float quant_scale, bool is_post_ln,
                                          bool out_col32) {
  int block_start = blockIdx.x * hidden_size;
  int start = block_start + threadIdx.x;
  int end = block_start + hidden_size;
  float val = 0.0;
  for (int i = start; i < end; i += blockDim.x) {
    val += input[i];
  }

  // step 0. compute mean
  __shared__ float s_mean;
  float reduce_res = blockReduceSum<float>(val);
  if (threadIdx.x == 0) s_mean = reduce_res / float(hidden_size);
  __syncthreads();

  // step 1. compute variance
  val = 0.0;
  for (int i = start; i < end; i += blockDim.x) {
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
  for (int i = start; i < end; i += blockDim.x) {
    val = input[i] - s_mean;
    output_f = val * s_var * __ldg(&scale[i - block_start]) +
               __ldg(&bias[i - block_start]);
    int8_t res = float2int8(output_f, quant_scale);
    if (out_col32) {
      int row_id = blockIdx.x;
      int col_id = i - block_start;
      int col32_index =
          row_major2flat_col32(row_id, col_id, gridDim.x, hidden_size);
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
__global__ void ker_norm_layer_resual_i8O<__half>(
    __half *input, int8_t *output, const __half *scale, const __half *bias,
    const __half *residual_bias, const int half_hidden_size, float quant_scale,
    bool is_post_ln, bool out_col32) {
  int block_start = blockIdx.x * half_hidden_size;
  int start = block_start + threadIdx.x;
  int end = blockIdx.x * half_hidden_size + half_hidden_size;
  half2 *pinput = (half2 *)input;
  char2 *poutput = (char2 *)output;
  const half2 *pscale = (const half2 *)scale;
  const half2 *pbias = (const half2 *)bias;
  const half2 *presidual_bias = (const half2 *)residual_bias;
  float mean_dim = float(half_hidden_size) * 2.f;

  float val = 0.0;
  // step 0. compute mean
  for (int i = start; i < end; i += blockDim.x) {
    float2 local_f2 = safe_half2_to_float2(pinput[i]);
    val += local_f2.x + local_f2.y;
  }
  __shared__ float s_mean;
  float reduce_res = blockReduceSum<float>(val);
  if (threadIdx.x == 0) s_mean = reduce_res / mean_dim;
  __syncthreads();

  // step 1. compute variance
  val = 0.0;
  for (int i = start; i < end; i += blockDim.x) {
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
  for (int i = start; i < end; i += blockDim.x) {
    float2 scale_val = __half22float2(__ldg(&pscale[i - block_start]));
    float2 bias_val = __half22float2(__ldg(&pbias[i - block_start]));
    float2 local_f2 = safe_half2_to_float2(pinput[i]);
    local_f2.x = (local_f2.x - s_mean) * s_var * scale_val.x + bias_val.x;
    local_f2.y = (local_f2.y - s_mean) * s_var * scale_val.y + bias_val.y;
    output_c2.x = float2int8(local_f2.x, quant_scale);
    output_c2.y = float2int8(local_f2.y, quant_scale);

    if (out_col32) {
      int row_id = blockIdx.x;
      int col_id = (i - block_start) * 2;
      int col32_index = row_major2flat_col32(row_id, col_id, gridDim.x,
                                             half_hidden_size * 2) >>
                        1;
      poutput[col32_index] = output_c2;
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
void ker_norm_layer_resual_i8O_launcher(int token_num, int hidden_size,
                                        cudaStream_t stream, T *input,
                                        int8_t *output, const T *scale,
                                        const T *bias, const T *residual_bias,
                                        const int max_thread_per_block,
                                        float quant_scale, bool is_post_ln,
                                        bool out_col32) {
  ker_norm_layer_resual_i8O<T><<<token_num, max_thread_per_block, 0, stream>>>(
      input, output, scale, bias, residual_bias, hidden_size, quant_scale,
      is_post_ln, out_col32);
}

template <>
void ker_norm_layer_resual_i8O_launcher<__half>(
    int token_num, int hidden_size, cudaStream_t stream, __half *input,
    int8_t *output, const __half *scale, const __half *bias,
    const __half *residual_bias, const int max_thread_per_block,
    float quant_scale, bool is_post_ln, bool out_col32) {
  ker_norm_layer_resual_i8O<__half>
      <<<token_num, max_thread_per_block, 0, stream>>>(
          input, output, scale, bias, residual_bias, hidden_size / 2,
          quant_scale, is_post_ln, out_col32);
}

template void ker_norm_layer_resual_i8O_launcher<float>(
    int token_num, int hidden_size, cudaStream_t stream, float *input,
    int8_t *output, const float *scale, const float *bias,
    const float *residual_bias, const int max_thread_per_block,
    float quant_scale, bool is_post_ln, bool out_col32);

template void ker_norm_layer_resual_i8O_launcher<__half>(
    int token_num, int hidden_size, cudaStream_t stream, __half *input,
    int8_t *output, const __half *scale, const __half *bias,
    const __half *residual_bias, const int max_thread_per_block,
    float quant_scale, bool is_post_ln, bool out_col32);

template <typename T>
__global__ void ker_residual_bias_ln_i32I_i8O(
    const int32_t *input, const T *scale, const T *bias, const T *residual_bias,
    int8_t *output, T *residual, int hidden_size, float dequant_scale,
    float quant_scale, bool is_post_ln, bool in_col32, bool out_col32,
    const T *colsum) {
  extern __shared__ float s_row_out[];

  int block_start = blockIdx.x * hidden_size;
  int start = block_start + threadIdx.x;
  int end = block_start + hidden_size;
  float val = 0.0;
  int input_index;
  for (int i = start; i < end; i += blockDim.x) {
    if (in_col32) {
      int row_id = blockIdx.x;
      int col_id = i - block_start;
      input_index =
          row_major2flat_col32(row_id, col_id, gridDim.x, hidden_size);
    } else {
      input_index = i;
    }
    float residual_out =
        __int2float_rn(input[input_index]) * dequant_scale + residual[i];
    if (colsum != nullptr) residual_out += __ldg(&colsum[i - block_start]);
    s_row_out[i - block_start] = residual_out;
    val += residual_out;
  }

  // step 0. compute mean
  __shared__ float s_mean;
  float reduce_res = blockReduceSum<float>(val);
  if (threadIdx.x == 0) s_mean = reduce_res / __int2float_rn(hidden_size);
  __syncthreads();

  // step 1. compute variance
  val = 0.0;
  for (int i = start; i < end; i += blockDim.x) {
    float tmp = s_row_out[i - block_start] - s_mean;
    val += tmp * tmp;
  }
  __shared__ float s_var;
  reduce_res = blockReduceSum(val);
  if (threadIdx.x == 0)
    s_var = rsqrtf(reduce_res / float(hidden_size) + epsilon);
  __syncthreads();

  float output_f;

  // step 2. layer norm
  for (int i = start; i < end; i += blockDim.x) {
    val = s_row_out[i - block_start] - s_mean;
    output_f = val * s_var * __ldg(&scale[i - block_start]) +
               __ldg(&bias[i - block_start]);
    int8_t res = float2int8(output_f, quant_scale);
    int output_index;
    if (out_col32) {
      int row_id = blockIdx.x;
      int col_id = i - block_start;
      output_index =
          row_major2flat_col32(row_id, col_id, gridDim.x, hidden_size);
    } else {
      output_index = i;
    }
    output[output_index] = res;
    T res_bias_val = (residual_bias == nullptr)
                         ? T{0.f}
                         : __ldg(&residual_bias[i - block_start]);
    if (is_post_ln) {
      residual[i] = output_f + res_bias_val;
    } else {
      residual[i] = s_row_out[i - block_start] + res_bias_val;
    }
  }
}

template <>
__global__ void ker_residual_bias_ln_i32I_i8O<half>(
    const int32_t *input, const half *scale, const half *bias,
    const half *residual_bias, int8_t *output, half *residual, int hidden_size,
    float dequant_scale, float quant_scale, bool is_post_ln, bool in_col32,
    bool out_col32, const half *colsum) {
  extern __shared__ float s_row_out[];

  int block_start = blockIdx.x * hidden_size;
  int start = block_start + threadIdx.x;
  int end = block_start + hidden_size;
  float val = 0.0;
  int input_index;
  for (int i = start; i < end; i += blockDim.x) {
    if (in_col32) {
      int row_id = blockIdx.x;
      int col_id = i - block_start;
      input_index =
          row_major2flat_col32(row_id, col_id, gridDim.x, hidden_size);
    } else {
      input_index = i;
    }
    float residual_out = __int2float_rn(input[input_index]) * dequant_scale +
                         safe_half_to_float(residual[i]);
    if (colsum != nullptr)
      residual_out += safe_half_to_float(__ldg(&colsum[i - block_start]));
    s_row_out[i - block_start] = residual_out;
    val += residual_out;
  }

  // step 0. compute mean
  __shared__ float s_mean;
  float reduce_res = blockReduceSum<float>(val);
  if (threadIdx.x == 0) s_mean = reduce_res / __int2float_rn(hidden_size);
  __syncthreads();

  // step 1. compute variance
  val = 0.0;
  for (int i = start; i < end; i += blockDim.x) {
    float tmp = s_row_out[i - block_start] - s_mean;
    val += tmp * tmp;
  }
  __shared__ float s_var;
  reduce_res = blockReduceSum(val);
  if (threadIdx.x == 0)
    s_var = rsqrtf(reduce_res / __int2float_rn(hidden_size) + epsilon);
  __syncthreads();

  float output_f;

  // step 2. layer norm
  for (int i = start; i < end; i += blockDim.x) {
    val = s_row_out[i - block_start] - s_mean;
    output_f =
        val * s_var * safe_half_to_float(__ldg(&scale[i - block_start])) +
        safe_half_to_float(__ldg(&bias[i - block_start]));
    int8_t res = float2int8(output_f, quant_scale);
    int output_index;
    if (out_col32) {
      int row_id = blockIdx.x;
      int col_id = i - block_start;
      output_index =
          row_major2flat_col32(row_id, col_id, gridDim.x, hidden_size);
    } else {
      output_index = i;
    }
    output[output_index] = res;

    half res_bias_val = (residual_bias == nullptr)
                            ? __float2half(0.f)
                            : __ldg(&residual_bias[i - block_start]);
    if (is_post_ln) {
      residual[i] = __float2half(output_f) + res_bias_val;
    } else {
      residual[i] = __float2half(s_row_out[i - block_start]) + res_bias_val;
    }
  }
}

template <typename T>
void ker_residual_bias_ln_i32I_i8O_launcher(
    const int32_t *input, const T *scale, const T *bias, const T *residual_bias,
    int8_t *output, T *residual, int batch_tokens, int hidden_size,
    float dequant_scale, float quant_scale, int max_thread_per_block,
    cudaStream_t stream, bool is_post_ln, bool in_col32, bool out_col32,
    const T *colsum) {
  ker_residual_bias_ln_i32I_i8O<T><<<batch_tokens, max_thread_per_block,
                                     hidden_size * sizeof(float), stream>>>(
      input, scale, bias, residual_bias, output, residual, hidden_size,
      dequant_scale, quant_scale, is_post_ln, in_col32, out_col32, colsum);
}

template <>
void ker_residual_bias_ln_i32I_i8O_launcher<half>(
    const int32_t *input, const half *scale, const half *bias,
    const half *residual_bias, int8_t *output, half *residual, int batch_tokens,
    int hidden_size, float dequant_scale, float quant_scale,
    int max_thread_per_block, cudaStream_t stream, bool is_post_ln,
    bool in_col32, bool out_col32, const half *colsum) {
  ker_residual_bias_ln_i32I_i8O<half><<<batch_tokens, max_thread_per_block,
                                        hidden_size * sizeof(float), stream>>>(
      input, scale, bias, residual_bias, output, residual, hidden_size,
      dequant_scale, quant_scale, is_post_ln, in_col32, out_col32, colsum);
}

template void ker_residual_bias_ln_i32I_i8O_launcher<float>(
    const int32_t *input, const float *scale, const float *bias,
    const float *residual_bias, int8_t *output, float *residual,
    int batch_tokens, int hidden_size, float dequant_scale, float quant_scale,
    int max_thread_per_block, cudaStream_t stream, bool is_post_ln,
    bool in_col32, bool out_col32, const float *colsum);

template void ker_residual_bias_ln_i32I_i8O_launcher<half>(
    const int32_t *input, const half *scale, const half *bias,
    const half *residual_bias, int8_t *output, half *residual, int batch_tokens,
    int hidden_size, float dequant_scale, float quant_scale,
    int max_thread_per_block, cudaStream_t stream, bool is_post_ln,
    bool in_col32, bool out_col32, const half *colsum);

template <typename T>
__global__ void ker_residual_bias_ln_i8I_i8O(
    const int8_t *input, const T *scale, const T *bias, const T *residual_bias,
    int8_t *output, T *residual, int hidden_size, float dequant_scale,
    float quant_scale, bool is_post_ln, bool in_col32, bool out_col32,
    const T *colsum) {
  extern __shared__ float s_row_out[];

  int block_start = blockIdx.x * hidden_size;
  int start = block_start + threadIdx.x;
  int end = block_start + hidden_size;
  float val = 0.0;
  int input_index;
  for (int i = start; i < end; i += blockDim.x) {
    if (in_col32) {
      int row_id = blockIdx.x;
      int col_id = i - block_start;
      input_index =
          row_major2flat_col32(row_id, col_id, gridDim.x, hidden_size);
    } else {
      input_index = i;
    }
    float residual_out =
        __int2float_rn(input[input_index]) * dequant_scale + residual[i];
    if (colsum) residual_out += colsum[i - block_start];
    s_row_out[i - block_start] = residual_out;
    val += residual_out;
  }

  // step 0. compute mean
  __shared__ float s_mean;
  float reduce_res = blockReduceSum<float>(val);
  if (threadIdx.x == 0) s_mean = reduce_res / __int2float_rn(hidden_size);
  __syncthreads();

  // step 1. compute variance
  val = 0.0;
  for (int i = start; i < end; i += blockDim.x) {
    float tmp = s_row_out[i - block_start] - s_mean;
    val += tmp * tmp;
  }
  __shared__ float s_var;
  reduce_res = blockReduceSum(val);
  if (threadIdx.x == 0)
    s_var = rsqrtf(reduce_res / float(hidden_size) + epsilon);
  __syncthreads();

  float output_f;

  // step 2. layer norm
  for (int i = start; i < end; i += blockDim.x) {
    val = s_row_out[i - block_start] - s_mean;
    output_f = val * s_var * __ldg(&scale[i - block_start]) +
               __ldg(&bias[i - block_start]);
    int8_t res = float2int8(output_f, quant_scale);
    int output_index;
    if (out_col32) {
      int row_id = blockIdx.x;
      int col_id = i - block_start;
      output_index =
          row_major2flat_col32(row_id, col_id, gridDim.x, hidden_size);
    } else {
      output_index = i;
    }
    output[output_index] = res;
    T res_bias_val = (residual_bias == nullptr)
                         ? T{0.f}
                         : __ldg(&residual_bias[i - block_start]);
    if (is_post_ln) {
      residual[i] = output_f + res_bias_val;
    } else {
      residual[i] = s_row_out[i - block_start] + res_bias_val;
    }
  }
}

template <>
__global__ void ker_residual_bias_ln_i8I_i8O<half>(
    const int8_t *input, const half *scale, const half *bias,
    const half *residual_bias, int8_t *output, half *residual, int hidden_size,
    float dequant_scale, float quant_scale, bool is_post_ln, bool in_col32,
    bool out_col32, const half *colsum) {
  extern __shared__ float s_row_out[];

  int block_start = blockIdx.x * hidden_size;
  int start = block_start + threadIdx.x;
  int end = block_start + hidden_size;
  float val = 0.0;
  int input_index;
  for (int i = start; i < end; i += blockDim.x) {
    if (in_col32) {
      int row_id = blockIdx.x;
      int col_id = i - block_start;
      input_index =
          row_major2flat_col32(row_id, col_id, gridDim.x, hidden_size);
    } else {
      input_index = i;
    }
    float residual_out = __int2float_rn(input[input_index]) * dequant_scale +
                         safe_half_to_float(residual[i]);
    if (colsum) residual_out += safe_half_to_float(colsum[i - block_start]);
    s_row_out[i - block_start] = residual_out;
    val += residual_out;
  }

  // step 0. compute mean
  __shared__ float s_mean;
  float reduce_res = blockReduceSum<float>(val);
  if (threadIdx.x == 0) s_mean = reduce_res / __int2float_rn(hidden_size);
  __syncthreads();

  // step 1. compute variance
  val = 0.0;
  for (int i = start; i < end; i += blockDim.x) {
    float tmp = s_row_out[i - block_start] - s_mean;
    val += tmp * tmp;
  }
  __shared__ float s_var;
  reduce_res = blockReduceSum(val);
  if (threadIdx.x == 0)
    s_var = rsqrtf(reduce_res / __int2float_rn(hidden_size) + epsilon);
  __syncthreads();

  float output_f;

  // step 2. layer norm
  for (int i = start; i < end; i += blockDim.x) {
    val = s_row_out[i - block_start] - s_mean;
    output_f =
        val * s_var * safe_half_to_float(__ldg(&scale[i - block_start])) +
        safe_half_to_float(__ldg(&bias[i - block_start]));
    int8_t res = float2int8(output_f, quant_scale);
    int output_index;
    if (out_col32) {
      int row_id = blockIdx.x;
      int col_id = i - block_start;
      output_index =
          row_major2flat_col32(row_id, col_id, gridDim.x, hidden_size);
    } else {
      output_index = i;
    }
    output[output_index] = res;

    half res_bias_val = (residual_bias == nullptr)
                            ? __float2half(0.f)
                            : __ldg(&residual_bias[i - block_start]);
    if (is_post_ln) {
      residual[i] = __float2half(output_f) + res_bias_val;
    } else {
      residual[i] = __float2half(s_row_out[i - block_start]) + res_bias_val;
    }
  }
}

template <typename T>
void ker_residual_bias_ln_i8I_i8O_launcher(
    const int8_t *input, const T *scale, const T *bias, const T *residual_bias,
    int8_t *output, T *residual, int batch_tokens, int hidden_size,
    float dequant_scale, float quant_scale, int max_thread_per_block,
    cudaStream_t stream, bool is_post_ln, bool in_col32, bool out_col32,
    const T *colsum) {
  ker_residual_bias_ln_i8I_i8O<T><<<batch_tokens, max_thread_per_block,
                                    hidden_size * sizeof(float), stream>>>(
      input, scale, bias, residual_bias, output, residual, hidden_size,
      dequant_scale, quant_scale, is_post_ln, in_col32, out_col32, colsum);
}

template <>
void ker_residual_bias_ln_i8I_i8O_launcher<half>(
    const int8_t *input, const half *scale, const half *bias,
    const half *residual_bias, int8_t *output, half *residual, int batch_tokens,
    int hidden_size, float dequant_scale, float quant_scale,
    int max_thread_per_block, cudaStream_t stream, bool is_post_ln,
    bool in_col32, bool out_col32, const half *colsum) {
  ker_residual_bias_ln_i8I_i8O<half><<<batch_tokens, max_thread_per_block,
                                       hidden_size * sizeof(float), stream>>>(
      input, scale, bias, residual_bias, output, residual, hidden_size,
      dequant_scale, quant_scale, is_post_ln, in_col32, out_col32, colsum);
}

template void ker_residual_bias_ln_i8I_i8O_launcher<float>(
    const int8_t *input, const float *scale, const float *bias,
    const float *residual_bias, int8_t *output, float *residual,
    int batch_tokens, int hidden_size, float dequant_scale, float quant_scale,
    int max_thread_per_block, cudaStream_t stream, bool is_post_ln,
    bool in_col32, bool out_col32, const float *colsum);

template void ker_residual_bias_ln_i8I_i8O_launcher<half>(
    const int8_t *input, const half *scale, const half *bias,
    const half *residual_bias, int8_t *output, half *residual, int batch_tokens,
    int hidden_size, float dequant_scale, float quant_scale,
    int max_thread_per_block, cudaStream_t stream, bool is_post_ln,
    bool in_col32, bool out_col32, const half *colsum);

template <typename T>
__global__ void ker_residual_bias_ln_i32I(const int32_t *input, const T *scale,
                                          const T *bias, const T *residual,
                                          T *output, int hidden_size,
                                          float dequant_scale, bool in_col32,
                                          const T *colsum) {
  extern __shared__ float s_row_out[];

  int block_start = blockIdx.x * hidden_size;
  int start = block_start + threadIdx.x;
  int end = block_start + hidden_size;
  float val = 0.0;
  int input_index;
  for (int i = start; i < end; i += blockDim.x) {
    if (in_col32) {
      int row_id = blockIdx.x;
      int col_id = i - block_start;
      input_index =
          row_major2flat_col32(row_id, col_id, gridDim.x, hidden_size);
    } else {
      input_index = i;
    }
    float residual_out =
        __int2float_rn(input[input_index]) * dequant_scale + residual[i];
    if (colsum) residual_out += colsum[i - block_start];

    s_row_out[i - block_start] = residual_out;
    val += residual_out;
  }

  // step 0. compute mean
  __shared__ float s_mean;
  float reduce_res = blockReduceSum<float>(val);
  if (threadIdx.x == 0) s_mean = reduce_res / __int2float_rn(hidden_size);
  __syncthreads();

  // step 1. compute variance
  val = 0.0;
  for (int i = start; i < end; i += blockDim.x) {
    float tmp = s_row_out[i - block_start] - s_mean;
    val += tmp * tmp;
  }
  __shared__ float s_var;
  reduce_res = blockReduceSum(val);
  if (threadIdx.x == 0)
    s_var = rsqrtf(reduce_res / float(hidden_size) + epsilon);
  __syncthreads();

  float output_f;

  // step 2. layer norm
  for (int i = start; i < end; i += blockDim.x) {
    val = s_row_out[i - block_start] - s_mean;
    output_f = val * s_var * __ldg(&scale[i - block_start]) +
               __ldg(&bias[i - block_start]);
    int output_index = i;
    output[output_index] = output_f;
  }
}

template <>
__global__ void ker_residual_bias_ln_i32I<half>(
    const int32_t *input, const half *scale, const half *bias,
    const half *residual, half *output, int hidden_size, float dequant_scale,
    bool in_col32, const half *colsum) {
  extern __shared__ float s_row_out[];

  int block_start = blockIdx.x * hidden_size;
  int start = block_start + threadIdx.x;
  int end = block_start + hidden_size;
  float val = 0.0;
  int input_index;
  for (int i = start; i < end; i += blockDim.x) {
    if (in_col32) {
      int row_id = blockIdx.x;
      int col_id = i - block_start;
      input_index =
          row_major2flat_col32(row_id, col_id, gridDim.x, hidden_size);
    } else {
      input_index = i;
    }
    float residual_out = __int2float_rn(input[input_index]) * dequant_scale +
                         safe_half_to_float(residual[i]);
    if (colsum) residual_out += safe_half_to_float(colsum[i - block_start]);

    s_row_out[i - block_start] = residual_out;
    val += residual_out;
  }

  // step 0. compute mean
  __shared__ float s_mean;
  float reduce_res = blockReduceSum<float>(val);
  if (threadIdx.x == 0) s_mean = reduce_res / __int2float_rn(hidden_size);
  __syncthreads();

  // step 1. compute variance
  val = 0.0;
  for (int i = start; i < end; i += blockDim.x) {
    float tmp = s_row_out[i - block_start] - s_mean;
    val += tmp * tmp;
  }
  __shared__ float s_var;
  reduce_res = blockReduceSum(val);
  if (threadIdx.x == 0)
    s_var = rsqrtf(reduce_res / __int2float_rn(hidden_size) + epsilon);
  __syncthreads();

  float output_f;

  // step 2. layer norm
  for (int i = start; i < end; i += blockDim.x) {
    val = s_row_out[i - block_start] - s_mean;
    output_f =
        val * s_var * safe_half_to_float(__ldg(&scale[i - block_start])) +
        safe_half_to_float(__ldg(&bias[i - block_start]));
    int output_index = i;
    output[output_index] = __float2half(output_f);
  }
}

template <typename T>
void ker_residual_bias_ln_i32I_launcher(const int32_t *input, const T *scale,
                                        const T *bias, const T *residual,
                                        T *output, int batch_tokens,
                                        int hidden_size, float dequant_scale,
                                        int max_thread_per_block,
                                        cudaStream_t stream, bool in_col32,
                                        const T *colsum) {
  ker_residual_bias_ln_i32I<T>
      <<<batch_tokens, max_thread_per_block, hidden_size * sizeof(float),
         stream>>>(input, scale, bias, residual, output, hidden_size,
                   dequant_scale, in_col32, colsum);
}

template <>
void ker_residual_bias_ln_i32I_launcher<half>(
    const int32_t *input, const half *scale, const half *bias,
    const half *residual, half *output, int batch_tokens, int hidden_size,
    float dequant_scale, int max_thread_per_block, cudaStream_t stream,
    bool in_col32, const half *colsum) {
  ker_residual_bias_ln_i32I<half>
      <<<batch_tokens, max_thread_per_block, hidden_size * sizeof(float),
         stream>>>(input, scale, bias, residual, output, hidden_size,
                   dequant_scale, in_col32, colsum);
}

template void ker_residual_bias_ln_i32I_launcher<float>(
    const int32_t *input, const float *scale, const float *bias,
    const float *residual, float *output, int batch_tokens, int hidden_size,
    float dequant_scale, int max_thread_per_block, cudaStream_t stream,
    bool in_col32, const float *colsum);

template void ker_residual_bias_ln_i32I_launcher<half>(
    const int32_t *input, const half *scale, const half *bias,
    const half *residual, half *output, int batch_tokens, int hidden_size,
    float dequant_scale, int max_thread_per_block, cudaStream_t stream,
    bool in_col32, const half *colsum);

template <typename T>
__global__ void ker_bias_gelu_i8I_i8O(int8_t *input, int8_t *output,
                                      const T *bias, int feature_dim,
                                      float dequant_scale, float quant_scale,
                                      bool in_col32, bool out_col32) {
  int block_start = blockIdx.x * feature_dim;
  int start = block_start + threadIdx.x;
  int end = block_start + feature_dim;
  for (int i = start; i < end; i += blockDim.x) {
    int input_index;
    if (in_col32) {
      int row_id = blockIdx.x;
      int col_id = i - block_start;
      input_index =
          row_major2flat_col32(row_id, col_id, gridDim.x, feature_dim);
    } else {
      input_index = i;
    }

    float fout = gelu<float>(float(input[input_index]) * dequant_scale +
                             __ldg(&bias[i - block_start]));

    int output_index;
    if (out_col32) {
      int row_id = blockIdx.x;
      int col_id = i - block_start;
      output_index =
          row_major2flat_col32(row_id, col_id, gridDim.x, feature_dim);
    } else {
      output_index = i;
    }
    output[output_index] = float2int8(fout, quant_scale);
  }
}

/* fp16 version */
template <>
__global__ void ker_bias_gelu_i8I_i8O<__half>(
    int8_t *input, int8_t *output, const __half *bias, int feature_dim,
    float dequant_scale, float quant_scale, bool in_col32, bool out_col32) {
  int block_start = blockIdx.x * feature_dim;
  int start = block_start + threadIdx.x;
  int end = block_start + feature_dim;
  for (int i = start; i < end; i += blockDim.x) {
    int input_index;
    if (in_col32) {
      int row_id = blockIdx.x;
      int col_id = i - block_start;
      input_index =
          row_major2flat_col32(row_id, col_id, gridDim.x, feature_dim);
    } else {
      input_index = i;
    }

    float fout = gelu<float>(float(input[input_index]) * dequant_scale +
                             __half2float(__ldg(&bias[i - block_start])));

    int output_index;
    if (out_col32) {
      int row_id = blockIdx.x;
      int col_id = i - block_start;
      output_index =
          row_major2flat_col32(row_id, col_id, gridDim.x, feature_dim);
    } else {
      output_index = i;
    }
    output[output_index] = float2int8(fout, quant_scale);
  }
}

template <typename T>
void ker_bias_gelu_i8I_i8O_launcher(int batch_token_num, cudaStream_t stream,
                                    int8_t *input, int8_t *output,
                                    const T *bias, int feature_dim,
                                    float dequant_scale, float quant_scale,
                                    bool in_col32, bool out_col32) {
  ker_bias_gelu_i8I_i8O<T><<<batch_token_num, 1024, 0, stream>>>(
      input, output, bias, feature_dim, dequant_scale, quant_scale, in_col32,
      out_col32);
}

template <>
void ker_bias_gelu_i8I_i8O_launcher<__half>(
    int batch_token_num, cudaStream_t stream, int8_t *input, int8_t *output,
    const __half *bias, int feature_dim, float dequant_scale, float quant_scale,
    bool in_col32, bool out_col32) {
  ker_bias_gelu_i8I_i8O<__half><<<batch_token_num, 1024, 0, stream>>>(
      input, output, bias, feature_dim, dequant_scale, quant_scale, in_col32,
      out_col32);
}

template void ker_bias_gelu_i8I_i8O_launcher<float>(
    int batch_token_num, cudaStream_t stream, int8_t *input, int8_t *output,
    const float *bias, int feature_dim, float dequant_scale, float quant_scale,
    bool in_col32, bool out_col32);

template void ker_bias_gelu_i8I_i8O_launcher<__half>(
    int batch_token_num, cudaStream_t stream, int8_t *input, int8_t *output,
    const __half *bias, int feature_dim, float dequant_scale, float quant_scale,
    bool in_col32, bool out_col32);

template <typename T>
__global__ void ker_bias_relu_i8I_i8O(int8_t *input, int8_t *output,
                                      const T *bias, int feature_dim,
                                      float dequant_scale, float quant_scale,
                                      float clip_max, bool in_col32,
                                      bool out_col32, bool narrow_clip) {
  int block_start = blockIdx.x * feature_dim;
  int start = block_start + threadIdx.x;
  int end = block_start + feature_dim;
  for (int i = start; i < end; i += blockDim.x) {
    int input_index;
    if (in_col32) {
      int row_id = blockIdx.x;
      int col_id = i - block_start;
      input_index =
          row_major2flat_col32(row_id, col_id, gridDim.x, feature_dim);
    } else {
      input_index = i;
    }

    float fout = fmaxf(float(input[input_index]) * dequant_scale +
                           __ldg(&bias[i - block_start]),
                       0.f);

    int output_index;
    if (out_col32) {
      int row_id = blockIdx.x;
      int col_id = i - block_start;
      output_index =
          row_major2flat_col32(row_id, col_id, gridDim.x, feature_dim);
    } else {
      output_index = i;
    }
    if (narrow_clip) {
      output[output_index] = posfloat2int8(fout, quant_scale, clip_max);
    } else {
      output[output_index] = float2int8(fout, quant_scale);
    }
  }
}

/* fp16 version */
template <>
__global__ void ker_bias_relu_i8I_i8O<__half>(
    int8_t *input, int8_t *output, const __half *bias, int feature_dim,
    float dequant_scale, float quant_scale, float clip_max, bool in_col32,
    bool out_col32, bool narrow_clip) {
  int block_start = blockIdx.x * feature_dim;
  int start = block_start + threadIdx.x;
  int end = block_start + feature_dim;
  for (int i = start; i < end; i += blockDim.x) {
    int input_index;
    if (in_col32) {
      int row_id = blockIdx.x;
      int col_id = i - block_start;
      input_index =
          row_major2flat_col32(row_id, col_id, gridDim.x, feature_dim);
    } else {
      input_index = i;
    }

    float fout = fmaxf(float(input[input_index]) * dequant_scale +
                           __half2float(__ldg(&bias[i - block_start])),
                       0.f);

    int output_index;
    if (out_col32) {
      int row_id = blockIdx.x;
      int col_id = i - block_start;
      output_index =
          row_major2flat_col32(row_id, col_id, gridDim.x, feature_dim);
    } else {
      output_index = i;
    }
    if (narrow_clip) {
      output[output_index] = posfloat2int8(fout, quant_scale, clip_max);
    } else {
      output[output_index] = float2int8(fout, quant_scale);
    }
  }
}

template <typename T>
void ker_bias_relu_i8I_i8O_launcher(int batch_token_num, cudaStream_t stream,
                                    int8_t *input, int8_t *output,
                                    const T *bias, int feature_dim,
                                    float dequant_scale, float quant_scale,
                                    float clip_max, bool in_col32,
                                    bool out_col32, bool narrow_clip) {
  ker_bias_relu_i8I_i8O<T><<<batch_token_num, 1024, 0, stream>>>(
      input, output, bias, feature_dim, dequant_scale, quant_scale, clip_max,
      in_col32, out_col32, narrow_clip);
}

template <>
void ker_bias_relu_i8I_i8O_launcher<__half>(
    int batch_token_num, cudaStream_t stream, int8_t *input, int8_t *output,
    const __half *bias, int feature_dim, float dequant_scale, float quant_scale,
    float clip_max, bool in_col32, bool out_col32, bool narrow_clip) {
  ker_bias_relu_i8I_i8O<__half><<<batch_token_num, 1024, 0, stream>>>(
      input, output, bias, feature_dim, dequant_scale, quant_scale, clip_max,
      in_col32, out_col32, narrow_clip);
}

template void ker_bias_relu_i8I_i8O_launcher<float>(
    int batch_token_num, cudaStream_t stream, int8_t *input, int8_t *output,
    const float *bias, int feature_dim, float dequant_scale, float quant_scale,
    float clip_max, bool in_col32, bool out_col32, bool narrow_clip);

template void ker_bias_relu_i8I_i8O_launcher<__half>(
    int batch_token_num, cudaStream_t stream, int8_t *input, int8_t *output,
    const __half *bias, int feature_dim, float dequant_scale, float quant_scale,
    float clip_max, bool in_col32, bool out_col32, bool narrow_clip);

template <typename T>
__global__ void ker_arrange_encself_qkv_i8I(const int8_t *ori_qkv,
                                            const T *qkv_bias, T *new_qkv,
                                            int max_batch_dim,
                                            int batch_seq_len, int dim_per_head,
                                            int head_num, float dequant_scale,
                                            bool in_col32) {
  int hidden_size = dim_per_head * head_num;
  int batch_id = blockIdx.x / batch_seq_len;
  int token_id = blockIdx.x % batch_seq_len;
  int qkv_offset = max_batch_dim * blockIdx.y;
  for (std::size_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    int target_id = targetid_4dim(batch_id, head_id, token_id, dim_id, head_num,
                                  batch_seq_len, dim_per_head);
    int qkv_index;
    if (in_col32) {
      int row_id = blockIdx.x;
      int col_id = blockIdx.y * hidden_size + i;
      qkv_index = row_major2flat_col32(row_id, col_id, gridDim.x,
                                       gridDim.y * hidden_size);
    } else {
      qkv_index = (blockIdx.x * gridDim.y + blockIdx.y) * hidden_size + i;
    }

    new_qkv[qkv_offset + target_id] =
        float(ori_qkv[qkv_index]) * dequant_scale +
        __ldg(&qkv_bias[blockIdx.y * hidden_size + i]);
  }
}

template <>
__global__ void ker_arrange_encself_qkv_i8I<__half>(
    const int8_t *ori_qkv, const __half *qkv_bias, __half *new_qkv,
    int max_batch_dim, int batch_seq_len, int dim_per_head, int head_num,
    float dequant_scale, bool in_col32) {
  int half_hidden_size = dim_per_head * head_num;
  int batch_id = blockIdx.x / batch_seq_len;
  int token_id = blockIdx.x % batch_seq_len;
  for (std::size_t i = threadIdx.x; i < half_hidden_size; i += blockDim.x) {
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    int qkv_offset = max_batch_dim * blockIdx.y;
    int target_id = targetid_4dim(batch_id, head_id, token_id, dim_id, head_num,
                                  batch_seq_len, dim_per_head);

    const char2 *p_ori_qkv = (const char2 *)ori_qkv;
    const half2 *p_bias = (const half2 *)qkv_bias;
    half2 *p_new_qkv = (half2 *)new_qkv;
    int qkv_index;
    if (in_col32) {
      int row_id = blockIdx.x;
      int col_id = (blockIdx.y * half_hidden_size + i) * 2;
      qkv_index = row_major2flat_col32(row_id, col_id, gridDim.x,
                                       gridDim.y * half_hidden_size * 2) >>
                  1;
    } else {
      qkv_index = (blockIdx.x * gridDim.y + blockIdx.y) * half_hidden_size + i;
    }
    char2 ori_qkv_i2 = p_ori_qkv[qkv_index];
    half2 ori_qkv_h2;
    ori_qkv_h2.x = __float2half(float(ori_qkv_i2.x) * dequant_scale);
    ori_qkv_h2.y = __float2half(float(ori_qkv_i2.y) * dequant_scale);
    p_new_qkv[qkv_offset + target_id] =
        __hadd2(ori_qkv_h2, __ldg(&p_bias[blockIdx.y * half_hidden_size + i]));
  }
}

template <typename T>
void ker_arrange_encself_qkv_i8I_launcher(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const int8_t *ori_qkv, const T *qkv_bias, T *new_qkv, int max_batch_dim,
    int batch_seq_len, int dim_per_head, int head_num, int max_thread_per_block,
    float dequant_scale, bool in_col32) {
  ker_arrange_encself_qkv_i8I<T>
      <<<dim3(batch_token_num, 3), max_thread_per_block, 0, stream>>>(
          ori_qkv, qkv_bias, new_qkv, max_batch_dim, batch_seq_len,
          dim_per_head, head_num, dequant_scale, in_col32);
}

template <>
void ker_arrange_encself_qkv_i8I_launcher<__half>(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const int8_t *ori_qkv, const __half *qkv_bias, __half *new_qkv,
    int max_batch_dim, int batch_seq_len, int dim_per_head, int head_num,
    int max_thread_per_block, float dequant_scale, bool in_col32) {
  ker_arrange_encself_qkv_i8I<__half>
      <<<dim3(batch_token_num, 3), max_thread_per_block, 0, stream>>>(
          ori_qkv, qkv_bias, new_qkv, max_batch_dim / 2, batch_seq_len,
          dim_per_head / 2, head_num, dequant_scale, in_col32);
}

template void ker_arrange_encself_qkv_i8I_launcher<float>(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const int8_t *ori_qkv, const float *qkv_bias, float *new_qkv,
    int max_batch_dim, int batch_seq_len, int dim_per_head, int head_num,
    int max_thread_per_block, float dequant_scale, bool in_col32);

template void ker_arrange_encself_qkv_i8I_launcher<__half>(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const int8_t *ori_qkv, const __half *qkv_bias, __half *new_qkv,
    int max_batch_dim, int batch_seq_len, int dim_per_head, int head_num,
    int max_thread_per_block, float dequant_scale, bool in_col32);

template <typename T>
__global__ void ker_arrange_encself_qkv_i8I_i8O(
    const int8_t *ori_qkv, const T *qkv_bias, int8_t *new_q, int8_t *new_k,
    int8_t *new_v, T *d_v, int batch_seq_len, int dim_per_head, int head_num,
    float dequant_scale, float quant_scale, bool in_col32) {
  int hidden_size = dim_per_head * head_num;
  int batch_id = blockIdx.x / batch_seq_len;
  int token_id = blockIdx.x % batch_seq_len;
  for (std::size_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    int target_id = targetid_4dim(batch_id, head_id, token_id, dim_id, head_num,
                                  batch_seq_len, dim_per_head);
    int qkv_index;
    if (in_col32) {
      int row_id = blockIdx.x;
      int col_id = blockIdx.y * hidden_size + i;
      qkv_index = row_major2flat_col32(row_id, col_id, gridDim.x,
                                       gridDim.y * hidden_size);
    } else {
      qkv_index = (blockIdx.x * gridDim.y + blockIdx.y) * hidden_size + i;
    }

    float val = float(ori_qkv[qkv_index]) * dequant_scale +
                __ldg(&qkv_bias[blockIdx.y * hidden_size + i]);
    int8_t quant_val = float2int8(val, quant_scale);

    if (blockIdx.y == 0) {
      new_q[target_id] = quant_val;
    } else if (blockIdx.y == 1) {
      new_k[target_id] = quant_val;
    } else {
      new_v[target_id] = quant_val;
      d_v[target_id] = float(quant_val) / quant_scale;
    }
  }
}

template <>
__global__ void ker_arrange_encself_qkv_i8I_i8O<__half>(
    const int8_t *ori_qkv, const __half *qkv_bias, int8_t *new_q, int8_t *new_k,
    int8_t *new_v, __half *d_v, int batch_seq_len, int dim_per_head,
    int head_num, float dequant_scale, float quant_scale, bool in_col32) {
  int hidden_size = dim_per_head * head_num;
  int batch_id = blockIdx.x / batch_seq_len;
  int token_id = blockIdx.x % batch_seq_len;
  for (std::size_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    int target_id = targetid_4dim(batch_id, head_id, token_id, dim_id, head_num,
                                  batch_seq_len, dim_per_head);
    int qkv_index;
    if (in_col32) {
      int row_id = blockIdx.x;
      int col_id = blockIdx.y * hidden_size + i;
      qkv_index = row_major2flat_col32(row_id, col_id, gridDim.x,
                                       gridDim.y * hidden_size);
    } else {
      qkv_index = (blockIdx.x * gridDim.y + blockIdx.y) * hidden_size + i;
    }

    float val = float(ori_qkv[qkv_index]) * dequant_scale +
                __half2float(__ldg(&qkv_bias[blockIdx.y * hidden_size + i]));
    int8_t quant_val = float2int8(val, quant_scale);

    if (blockIdx.y == 0) {
      new_q[target_id] = quant_val;
    } else if (blockIdx.y == 1) {
      new_k[target_id] = quant_val;
    } else {
      new_v[target_id] = quant_val;
      d_v[target_id] = __float2half(float(quant_val) / quant_scale);
    }
  }
}

template <typename T>
void ker_arrange_encself_qkv_i8I_i8O_launcher(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const int8_t *ori_qkv, const T *qkv_bias, int8_t *new_q, int8_t *new_k,
    int8_t *new_v, T *d_v, int batch_seq_len, int dim_per_head, int head_num,
    int max_thread_per_block, float dequant_scale, float quant_scale,
    bool in_col32) {
  ker_arrange_encself_qkv_i8I_i8O<T>
      <<<dim3(batch_token_num, 3), max_thread_per_block, 0, stream>>>(
          ori_qkv, qkv_bias, new_q, new_k, new_v, d_v, batch_seq_len,
          dim_per_head, head_num, dequant_scale, quant_scale, in_col32);
}

template <>
void ker_arrange_encself_qkv_i8I_i8O_launcher<__half>(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const int8_t *ori_qkv, const __half *qkv_bias, int8_t *new_q, int8_t *new_k,
    int8_t *new_v, __half *d_v, int batch_seq_len, int dim_per_head,
    int head_num, int max_thread_per_block, float dequant_scale,
    float quant_scale, bool in_col32) {
  ker_arrange_encself_qkv_i8I_i8O<__half>
      <<<dim3(batch_token_num, 3), max_thread_per_block, 0, stream>>>(
          ori_qkv, qkv_bias, new_q, new_k, new_v, d_v, batch_seq_len,
          dim_per_head, head_num, dequant_scale, quant_scale, in_col32);
}

template void ker_arrange_encself_qkv_i8I_i8O_launcher<float>(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const int8_t *ori_qkv, const float *qkv_bias, int8_t *new_q, int8_t *new_k,
    int8_t *new_v, float *d_v, int batch_seq_len, int dim_per_head,
    int head_num, int max_thread_per_block, float dequant_scale,
    float quant_scale, bool in_col32);

template void ker_arrange_encself_qkv_i8I_i8O_launcher<__half>(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const int8_t *ori_qkv, const __half *qkv_bias, int8_t *new_q, int8_t *new_k,
    int8_t *new_v, __half *d_v, int batch_seq_len, int dim_per_head,
    int head_num, int max_thread_per_block, float dequant_scale,
    float quant_scale, bool in_col32);

template <typename T>
__global__ void ker_arrange_atten_output_i8O(const T *ori_q, int8_t *new_q,
                                             int beam_size, int dim_per_head,
                                             int head_num, float quant_scale,
                                             bool out_col32) {
  int hidden_size = dim_per_head * head_num;
  int batch_id = blockIdx.x / beam_size;
  // note, for encoder, beam_id is token_id; for decoder, beam_id is beam_id
  int beam_id = blockIdx.x % beam_size;
  for (std::size_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    int out_index;
    if (out_col32) {
      int row_id = blockIdx.x;
      int col_id = i;
      out_index = row_major2flat_col32(row_id, col_id, gridDim.x, hidden_size);
    } else {
      out_index = blockIdx.x * hidden_size + i;
    }
    new_q[out_index] =
        float2int8(ori_q[targetid_4dim(batch_id, head_id, beam_id, dim_id,
                                       head_num, beam_size, dim_per_head)],
                   quant_scale);
  }
}

template <>
__global__ void ker_arrange_atten_output_i8O<__half>(
    const __half *ori_q, int8_t *new_q, int beam_size, int dim_per_head,
    int head_num, float quant_scale, bool out_col32) {
  int batch_id = blockIdx.x / beam_size;
  // note, for encoder, beam_id is token_id; for decoder, beam_id is beam_id
  int beam_id = blockIdx.x % beam_size;
  int half_hidden_size = dim_per_head * head_num;
  for (std::size_t i = threadIdx.x; i < half_hidden_size; i += blockDim.x) {
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    int out_index;
    if (out_col32) {
      int row_id = blockIdx.x;
      int col_id = i * 2;
      out_index = row_major2flat_col32(row_id, col_id, gridDim.x,
                                       half_hidden_size * 2) >>
                  1;
    } else {
      out_index = blockIdx.x * half_hidden_size + i;
    }
    const half2 *p_ori_q = (const half2 *)ori_q;
    half2 v_ori_q;
    char2 *p_new_q = (char2 *)new_q;
    char2 v_new_q;
    v_ori_q = p_ori_q[targetid_4dim(batch_id, head_id, beam_id, dim_id,
                                    head_num, beam_size, dim_per_head)];
    v_new_q.x = float2int8(float(v_ori_q.x), quant_scale);
    v_new_q.y = float2int8(float(v_ori_q.y), quant_scale);
    p_new_q[out_index] = v_new_q;
  }
}

template <typename T>
void ker_arrange_atten_output_i8O_launcher(int batch_token_num, int hidden_size,
                                           cudaStream_t stream, const T *ori_q,
                                           int8_t *new_q, int beam_size,
                                           int dim_per_head, int head_num,
                                           int max_thread_per_block,
                                           float quant_scale, bool out_col32) {
  ker_arrange_atten_output_i8O<T>
      <<<batch_token_num, max_thread_per_block, 0, stream>>>(
          ori_q, new_q, beam_size, dim_per_head, head_num, quant_scale,
          out_col32);
}

template <>
void ker_arrange_atten_output_i8O_launcher<__half>(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const __half *ori_q, int8_t *new_q, int beam_size, int dim_per_head,
    int head_num, int max_thread_per_block, float quant_scale, bool out_col32) {
  ker_arrange_atten_output_i8O<__half>
      <<<batch_token_num, max_thread_per_block, 0, stream>>>(
          ori_q, new_q, beam_size, dim_per_head / 2, head_num, quant_scale,
          out_col32);
}

template void ker_arrange_atten_output_i8O_launcher<float>(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const float *ori_q, int8_t *new_q, int beam_size, int dim_per_head,
    int head_num, int max_thread_per_block, float quant_scale, bool out_col32);

template void ker_arrange_atten_output_i8O_launcher<__half>(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const __half *ori_q, int8_t *new_q, int beam_size, int dim_per_head,
    int head_num, int max_thread_per_block, float quant_scale, bool out_col32);

template <typename T>
__global__ void ker_arrange_decself_qkv_i8I_i8O(
    const int8_t *ori_qkv, const T *qkv_bias, int8_t *new_q, int8_t *new_k,
    int8_t *new_v, int head_num, int dim_per_head, int max_step, int step_id,
    float dequant_scale, float quant_scale, bool in_col32) {
  int hidden_size = dim_per_head * head_num;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    int qkv_index;
    if (in_col32) {
      int row_id = blockIdx.x;
      int col_id = blockIdx.y * hidden_size + i;
      qkv_index = row_major2flat_col32(row_id, col_id, gridDim.x,
                                       gridDim.y * hidden_size);
    } else {
      qkv_index = (blockIdx.x * gridDim.y + blockIdx.y) * hidden_size + i;
    }
    float val = float(ori_qkv[qkv_index]) * dequant_scale +
                __ldg(&qkv_bias[blockIdx.y * hidden_size + i]);
    int8_t quant_val = float2int8(val, quant_scale);
    int seq_id =
        blockIdx.x;  // obvious, seq_id = batch_id * beam_size + beam_id
    if (blockIdx.y == 0) {
      // for query
      new_q[seq_id * hidden_size + i] = quant_val;
      return;
    }
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    int target_id = targetid_4dim(seq_id, head_id, step_id, dim_id, head_num,
                                  max_step, dim_per_head);
    if (blockIdx.y == 1) {
      // for key
      new_k[target_id] = quant_val;
    } else {
      // for value
      new_v[target_id] = quant_val;
    }
  }
}

template <>
__global__ void ker_arrange_decself_qkv_i8I_i8O<__half>(
    const int8_t *ori_qkv, const __half *qkv_bias, int8_t *new_q, int8_t *new_k,
    int8_t *new_v, int head_num, int dim_per_head, int max_step, int step_id,
    float dequant_scale, float quant_scale, bool in_col32) {
  int hidden_size = dim_per_head * head_num;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    int qkv_index;
    if (in_col32) {
      int row_id = blockIdx.x;
      int col_id = blockIdx.y * hidden_size + i;
      qkv_index = row_major2flat_col32(row_id, col_id, gridDim.x,
                                       gridDim.y * hidden_size);
    } else {
      qkv_index = (blockIdx.x * gridDim.y + blockIdx.y) * hidden_size + i;
    }
    float val = float(ori_qkv[qkv_index]) * dequant_scale +
                __half2float(__ldg(&qkv_bias[blockIdx.y * hidden_size + i]));
    int8_t quant_val = float2int8(val, quant_scale);
    int seq_id =
        blockIdx.x;  // obvious, seq_id = batch_id * beam_size + beam_id
    if (blockIdx.y == 0) {
      // for query
      new_q[seq_id * hidden_size + i] = quant_val;
      return;
    }
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    int target_id = targetid_4dim(seq_id, head_id, step_id, dim_id, head_num,
                                  max_step, dim_per_head);
    if (blockIdx.y == 1) {
      // for key
      new_k[target_id] = quant_val;
    } else {
      // for value
      new_v[target_id] = quant_val;
    }
  }
}

template <typename T>
void ker_arrange_decself_qkv_i8I_i8O_launcher(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const int8_t *ori_qkv, const T *qkv_bias, int8_t *new_q, int8_t *new_k,
    int8_t *new_v, int head_num, int dim_per_head, int max_step, int step_id,
    int max_thread_per_block, float dequant_scale, float quant_scale,
    bool in_col32) {
  ker_arrange_decself_qkv_i8I_i8O<T>
      <<<dim3(step_token_num, 3), max_thread_per_block, 0, stream>>>(
          ori_qkv, qkv_bias, new_q, new_k, new_v, head_num, dim_per_head,
          max_step, step_id, dequant_scale, quant_scale, in_col32);
}

// template <>
// void ker_arrange_decself_qkv_i8I_i8O_launcher<__half>(
//     int step_token_num, int hidden_size, cudaStream_t stream,
//     const int8_t *ori_qkv, const __half *qkv_bias, int8_t *new_q, int8_t
//     *new_k, int8_t *new_v, int head_num, int dim_per_head, int max_step, int
//     step_id, int max_thread_per_block, float dequant_scale, float
//     quant_scale, bool in_col32) {
//   ker_arrange_decself_qkv_i8I_i8O<__half>
//       <<<dim3(step_token_num, 3), max_thread_per_block, 0, stream>>>(
//           ori_qkv, qkv_bias, new_q, new_k, new_v, head_num, dim_per_head,
//           max_step, step_id, dequant_scale, quant_scale, in_col32);
// }

template void ker_arrange_decself_qkv_i8I_i8O_launcher<float>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const int8_t *ori_qkv, const float *qkv_bias, int8_t *new_q, int8_t *new_k,
    int8_t *new_v, int head_num, int dim_per_head, int max_step, int step_id,
    int max_thread_per_block, float dequant_scale, float quant_scale,
    bool in_col32);

template void ker_arrange_decself_qkv_i8I_i8O_launcher<__half>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const int8_t *ori_qkv, const __half *qkv_bias, int8_t *new_q, int8_t *new_k,
    int8_t *new_v, int head_num, int dim_per_head, int max_step, int step_id,
    int max_thread_per_block, float dequant_scale, float quant_scale,
    bool in_col32);

/**
@brief: ker_fuse_softmax_new_value_i32I_i8O
fused query-key correlation softmax and new_value for decoder self attention

@thread
gridDim.x = batch_size * beam_size * head_num
blockDim.x = first multiple of WARP_SIZE greater than cur_step + 1

@param
correlation: [batch_size, beam_size, head_num, cur_step + 1]
*/
__global__ void ker_fuse_softmax_new_value_i32I_i8O(
    const int32_t *logits, const int8_t *v, int8_t *new_v, int step_num,
    int max_step, int head_num, int dim_per_head, float attn_scale,
    float dequant_scale, float quant_scale, bool out_col32) {
  int idx = blockIdx.x * max_step + threadIdx.x;
  float val = threadIdx.x < step_num ? float(logits[idx]) * dequant_scale *
                                           dequant_scale * attn_scale
                                     : CUDA_FLOAT_INF_NEG;

  float max_val = blockReduceMax(val);
  __shared__ float smax;
  if (threadIdx.x == 0) smax = max_val;
  __syncthreads();

  val = threadIdx.x < step_num ? expf(val - smax) : 0;

  float rsum = blockReduceSum(val);
  __shared__ float ssum;
  if (threadIdx.x == 0) ssum = rsum;
  __syncthreads();

  extern __shared__ float block_new_value[];
  float *step_probs = &block_new_value[dim_per_head];
  if (threadIdx.x < step_num) step_probs[threadIdx.x] = val / ssum;
  __syncthreads();

  for (int i = threadIdx.x, end = step_num * dim_per_head; i < end;
       i += blockDim.x) {
    int value_idx = blockIdx.x * max_step * dim_per_head + i;
    int step_idx = i / dim_per_head;
    int dim_idx = i % dim_per_head;
    if (step_idx == 0) {
      block_new_value[dim_idx] = 0;
    }
    atomicAdd(&block_new_value[dim_idx],
              float(v[value_idx]) * step_probs[step_idx] * dequant_scale);
  }
  __syncthreads();

  for (int i = threadIdx.x, end = dim_per_head; i < end; i += blockDim.x) {
    int row = blockIdx.x / head_num;
    int head_idx = blockIdx.x % head_num;
    int row_size = gridDim.x / head_num;
    int col = head_idx * dim_per_head + i;
    int col_size = head_num * dim_per_head;
    int new_v_idx = row * col_size + col;
    if (out_col32) {
      new_v_idx = row_major2flat_col32(row, col, row_size, col_size);
    }
    new_v[new_v_idx] = float2int8(block_new_value[i], quant_scale);
  }
}

void ker_fuse_softmax_new_value_i32I_i8O_launcher(
    const int32_t *correlation, const int8_t *v, int8_t *new_v,
    int batch_head_num, int step_num, int max_step, int head_num,
    int dim_per_head, float attn_scale, float dequant_scale, float quant_scale,
    bool out_col32, cudaStream_t stream) {
  int block_dim = step_num;
  if (step_num < 1024) {
    block_dim = (step_num + 31) >> 5;
    block_dim *= 32;
  }
  ker_fuse_softmax_new_value_i32I_i8O<<<
      batch_head_num, block_dim,
      dim_per_head * sizeof(float) + step_num * sizeof(float), stream>>>(
      correlation, v, new_v, step_num, max_step, head_num, dim_per_head,
      attn_scale, dequant_scale, quant_scale, out_col32);
}

template <typename T>
__global__ void ker_arrange_encdec_q_i8I(const int8_t *ori_q, const T *q_bias,
                                         T *new_q, int beam_size,
                                         int dim_per_head, int head_num,
                                         float dequant_scale, bool in_col32) {
  int hidden_size = dim_per_head * head_num;
  for (std::size_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    int qkv_index;
    if (in_col32) {
      int row_id = blockIdx.x;
      int col_id = i;
      qkv_index = row_major2flat_col32(row_id, col_id, gridDim.x, hidden_size);
    } else {
      qkv_index = blockIdx.x * hidden_size + i;
    }
    T val = float(ori_q[qkv_index]) * dequant_scale + __ldg(&q_bias[i]);
    int batch_id = blockIdx.x / beam_size;
    int beam_id = blockIdx.x % beam_size;
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    new_q[targetid_4dim(batch_id, head_id, beam_id, dim_id, head_num, beam_size,
                        dim_per_head)] = val;
  }
}

template <>
__global__ void ker_arrange_encdec_q_i8I<__half>(
    const int8_t *ori_q, const __half *q_bias, __half *new_q, int beam_size,
    int dim_per_head, int head_num, float dequant_scale, bool in_col32) {
  int half_hidden_size = dim_per_head * head_num;
  const char2 *p_q = reinterpret_cast<const char2 *>(ori_q);
  for (std::size_t i = threadIdx.x; i < half_hidden_size; i += blockDim.x) {
    int qkv_index;
    if (in_col32) {
      int row_id = blockIdx.x;
      int col_id = i * 2;
      qkv_index = row_major2flat_col32(row_id, col_id, gridDim.x,
                                       half_hidden_size * 2) >>
                  1;
    } else {
      qkv_index = blockIdx.x * half_hidden_size + i;
    }
    char2 p_q_i2 = p_q[qkv_index];
    half2 p_q_h2;
    p_q_h2.x = __float2half(float(p_q_i2.x) * dequant_scale);
    p_q_h2.y = __float2half(float(p_q_i2.y) * dequant_scale);
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
void ker_arrange_encdec_q_i8I_launcher(int step_token_num, int hidden_size,
                                       cudaStream_t stream, const int8_t *ori_q,
                                       const T *q_bias, T *new_q, int beam_size,
                                       int dim_per_head, int head_num,
                                       int max_thread_per_block,
                                       float dequant_scale, bool in_col32) {
  ker_arrange_encdec_q_i8I<T>
      <<<step_token_num, max_thread_per_block, 0, stream>>>(
          ori_q, q_bias, new_q, beam_size, dim_per_head, head_num,
          dequant_scale, in_col32);
}

template <>
void ker_arrange_encdec_q_i8I_launcher<__half>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const int8_t *ori_q, const __half *q_bias, __half *new_q, int beam_size,
    int dim_per_head, int head_num, int max_thread_per_block,
    float dequant_scale, bool in_col32) {
  ker_arrange_encdec_q_i8I<__half>
      <<<step_token_num, max_thread_per_block, 0, stream>>>(
          ori_q, q_bias, new_q, beam_size, dim_per_head / 2, head_num,
          dequant_scale, in_col32);
}

template void ker_arrange_encdec_q_i8I_launcher<float>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const int8_t *ori_q, const float *q_bias, float *new_q, int beam_size,
    int dim_per_head, int head_num, int max_thread_per_block,
    float dequant_scale, bool in_col32);

template void ker_arrange_encdec_q_i8I_launcher<__half>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const int8_t *ori_q, const __half *q_bias, __half *new_q, int beam_size,
    int dim_per_head, int head_num, int max_thread_per_block,
    float dequant_scale, bool in_col32);

template <typename T, int beam_size>
__global__ void select_beam_rough_topk_i8I(
    const int8_t *logits, const T *logit_bias, const float *seq_probs,
    const float *seq_score, const int *alive_seq, float dequant_scale,
    int *can_idx, float *can_score, int *num_beam_can, int vocab_size,
    int max_step, float length_norm, int cur_step, float diverse_lambda,
    int end_id, bool in_col32) {
  if (cur_step != 0 && alive_seq[blockIdx.x * max_step + cur_step] == end_id) {
    // this is a finished beam
    if (threadIdx.x == 0) {
      num_beam_can[blockIdx.x + 1] = 1;      // generate one candidate
      int pos = atomicAdd(num_beam_can, 1);  // get a candidate pos
      if (diverse_lambda == 0) {
        can_score[pos] =
            seq_score[blockIdx.x];  // this beam's score will not be change
      } else {
        // add the beam id offset in score to sort in each beam
        int batch_id = blockIdx.x / beam_size;
        can_score[pos] = seq_score[blockIdx.x] +
                         (blockIdx.x - batch_id) * min_log_probability;
      }
      can_idx[pos] = end_id + (blockIdx.x % beam_size) * vocab_size;  // EOS
    }
    return;
  }

  /* step1: compute each thread's max_logit and sum_exp_logit, store in
   * rough_top_kth_logit, sum_exp_logit */
  const int block_start = blockIdx.x * vocab_size;
  const int left_idx = block_start + threadIdx.x;
  const int right_idx = (blockIdx.x + 1) * vocab_size;
  float rough_top_kth_logit = CUDA_FLOAT_INF_NEG;
  float sum_exp_logit = 0;
  for (int i = left_idx; i < right_idx; i += blockDim.x) {
    int logits_idx;
    if (in_col32) {
      int row_id = blockIdx.x;
      int col_id = i - block_start;
      logits_idx = row_major2flat_col32(row_id, col_id, gridDim.x, vocab_size);
    } else {
      logits_idx = i;
    }

    float lgt = (float)logits[logits_idx] * dequant_scale +
                (float)__ldg(&logit_bias[i - block_start]);
    rough_top_kth_logit = fmaxf(rough_top_kth_logit, lgt);
  }
  float max_logit = blockReduceMax(rough_top_kth_logit);
  __shared__ float s_max_logit;
  if (threadIdx.x == 0) {
    s_max_logit = max_logit;
  }
  __syncthreads();
  for (int i = left_idx; i < right_idx; i += blockDim.x) {
    int logits_idx;
    if (in_col32) {
      int row_id = blockIdx.x;
      int col_id = i - block_start;
      logits_idx = row_major2flat_col32(row_id, col_id, gridDim.x, vocab_size);
    } else {
      logits_idx = i;
    }

    float lgt =
        fmaxf((float)(logits[logits_idx]) * dequant_scale +
                  (float)__ldg(&logit_bias[i - block_start]) - s_max_logit,
              logit_thresh_min);
    sum_exp_logit += expf(lgt);
  }

  /*
  step2: compute rough top-kth-logits and sum_exp_logit among the whole beam,
  saved into s_topk and
      s_log_prob_base
  */
  __shared__ float
      s_log_prob_base;      // prefix sequence log prob - log_sum_exp_logit
  __shared__ float s_topk;  // rough top k-th value of logits
  __shared__ int num_cur_beam_can;  // candidate number for this beam
  sum_exp_logit = blockReduceSum(sum_exp_logit);
  rough_top_kth_logit = blockRoughTopK<float, beam_size>(rough_top_kth_logit);
  if (threadIdx.x == 0) {
    s_log_prob_base = seq_probs[blockIdx.x] - logf(sum_exp_logit) - s_max_logit;
    s_topk = rough_top_kth_logit;
    num_cur_beam_can = 0;
  }

  /*
  step3 : select the candidate token with logits bigger than s_topk,
          compute the seq probability ended with them,
      save the probability, token_index, selected token number.
  */
  int idx = left_idx;
  int batch_id = blockIdx.x / beam_size;
  int batch_start_pos = batch_id * beam_size * vocab_size;
  // int unk_vocab_id = vocab_size - 3;  // last three element: unk, start,
  // eos
  __shared__ int l_n;  // current iteration candidate number
  for (int iter = 0; iter < (vocab_size + blockDim.x - 1) / blockDim.x;
       iter++) {
    // zero the counter
    if (threadIdx.x == 0) l_n = 0;
    __syncthreads();

    float lgt = CUDA_FLOAT_INF_NEG - 1.f;  // min s_topk is CUDA_FLOAT_INF_NEG
    int pos;
    int vocab_id = idx - block_start;

    int logits_idx;
    if (in_col32) {
      int row_id = blockIdx.x;
      int col_id = vocab_id;
      logits_idx = row_major2flat_col32(row_id, col_id, gridDim.x, vocab_size);
    } else {
      logits_idx = idx;
    }

    // if ((vocab_id < vocab_size) && (vocab_id != unk_vocab_id)) {
    if (vocab_id < vocab_size) {
      lgt = (float)(logits[logits_idx]) * dequant_scale +
            (float)__ldg(&logit_bias[vocab_id]);
      if (lgt >= s_topk)
        // pos: relative pos inside this iteration
        pos = atomicAdd(&l_n, 1);
    }
    __syncthreads();

    // leader increments the global counter
    if (threadIdx.x == 0) {
      atomicAdd(&num_cur_beam_can, l_n);
      l_n = atomicAdd(num_beam_can, l_n);
    }
    __syncthreads();

    // threads with true predicates write their elements
    if ((lgt >= s_topk)) {
      pos += l_n;  // increment local pos by global counter
      if (diverse_lambda == 0) {
        can_score[pos] = fmaxf((lgt + s_log_prob_base) * length_norm,
                               min_log_probability + 1.f) +
                         batch_id * min_log_probability;
      } else {
        can_score[pos] = fmaxf((lgt + s_log_prob_base) * length_norm,
                               min_log_probability + 1.f) +
                         blockIdx.x * min_log_probability;
      }
      can_idx[pos] = idx - batch_start_pos;
    }
    __syncthreads();
    idx += blockDim.x;
  }
  if (threadIdx.x == 0) {
    num_beam_can[blockIdx.x + 1] = num_cur_beam_can;
  }
}

template <typename T>
void select_beam_rough_topk_i8I_launcher(
    const int8_t *logits, const T *logit_bias, const float *seq_probs,
    const float *seq_score, const int *alive_seq, float dequant_scale,
    int *can_idx, float *can_score, int *num_beam_can, int vocab_size,
    int max_step, float length_norm, int cur_step, int step_token_num,
    int max_thread_per_block, cudaStream_t stream, int beam_size,
    float diverse_lambda, int end_id, bool in_col32) {
  if (beam_size == 1)
    select_beam_rough_topk_i8I<T, 1>
        <<<step_token_num, max_thread_per_block, 0, stream>>>(
            logits, logit_bias, seq_probs, seq_score, alive_seq, dequant_scale,
            can_idx, can_score, num_beam_can, vocab_size, max_step, length_norm,
            cur_step, diverse_lambda, end_id, in_col32);
  if (beam_size == 2)
    select_beam_rough_topk_i8I<T, 2>
        <<<step_token_num, max_thread_per_block, 0, stream>>>(
            logits, logit_bias, seq_probs, seq_score, alive_seq, dequant_scale,
            can_idx, can_score, num_beam_can, vocab_size, max_step, length_norm,
            cur_step, diverse_lambda, end_id, in_col32);
  if (beam_size == 4)
    select_beam_rough_topk_i8I<T, 4>
        <<<step_token_num, max_thread_per_block, 0, stream>>>(
            logits, logit_bias, seq_probs, seq_score, alive_seq, dequant_scale,
            can_idx, can_score, num_beam_can, vocab_size, max_step, length_norm,
            cur_step, diverse_lambda, end_id, in_col32);
  if (beam_size == 8)
    select_beam_rough_topk_i8I<T, 8>
        <<<step_token_num, max_thread_per_block, 0, stream>>>(
            logits, logit_bias, seq_probs, seq_score, alive_seq, dequant_scale,
            can_idx, can_score, num_beam_can, vocab_size, max_step, length_norm,
            cur_step, diverse_lambda, end_id, in_col32);
  if (beam_size == 16)
    select_beam_rough_topk_i8I<T, 16>
        <<<step_token_num, max_thread_per_block, 0, stream>>>(
            logits, logit_bias, seq_probs, seq_score, alive_seq, dequant_scale,
            can_idx, can_score, num_beam_can, vocab_size, max_step, length_norm,
            cur_step, diverse_lambda, end_id, in_col32);
  if (beam_size == 32)
    select_beam_rough_topk_i8I<T, 32>
        <<<step_token_num, max_thread_per_block, 0, stream>>>(
            logits, logit_bias, seq_probs, seq_score, alive_seq, dequant_scale,
            can_idx, can_score, num_beam_can, vocab_size, max_step, length_norm,
            cur_step, diverse_lambda, end_id, in_col32);
}

template void select_beam_rough_topk_i8I_launcher<float>(
    const int8_t *logits, const float *logit_bias, const float *seq_probs,
    const float *seq_score, const int *alive_seq, float dequant_scale,
    int *can_idx, float *can_score, int *num_beam_can, int vocab_size,
    int max_step, float length_norm, int cur_step, int step_token_num,
    int max_thread_per_block, cudaStream_t stream, int beam_size,
    float diverse_lambda, int end_id, bool in_col32);

template void select_beam_rough_topk_i8I_launcher<__half>(
    const int8_t *logits, const __half *logit_bias, const float *seq_probs,
    const float *seq_score, const int *alive_seq, float dequant_scale,
    int *can_idx, float *can_score, int *num_beam_can, int vocab_size,
    int max_step, float length_norm, int cur_step, int step_token_num,
    int max_thread_per_block, cudaStream_t stream, int beam_size,
    float diverse_lambda, int end_id, bool in_col32);

template <typename T, int k>
__global__ void ker_topk_sample_i8I(const int8_t *logits, const T *logit_bias,
                                    int *old_input_ids, int *new_input_ids,
                                    const int vocab_size, const int max_step,
                                    const int batch_seq_len, int logits_seq_len,
                                    int *unfinished, curandState *curandstate,
                                    int eos_id, float dequant_scale,
                                    bool in_col32) {
  int last_token_idx_in_batch = blockIdx.x * max_step + batch_seq_len - 1;

  /* add EOS to end if last token is EOS */
  if (batch_seq_len > 1 && old_input_ids[last_token_idx_in_batch] == eos_id) {
    if (threadIdx.x == 0) {
      old_input_ids[last_token_idx_in_batch + 1] = eos_id;
    }
    return;
  }
  int logits_token_idx_in_batch =
      blockIdx.x * logits_seq_len + logits_seq_len - 1;
  int left_logit_idx = logits_token_idx_in_batch * vocab_size + threadIdx.x;
  int right_logit_idx = (logits_token_idx_in_batch + 1) * vocab_size;

  /*
  step1. find max logit and rough Kth logit over the whole vocab
  */
  __shared__ float s_max_logit, s_topk_logit;
  float rough_top_kth_logit = CUDA_FLOAT_INF_NEG;
  for (int idx = left_logit_idx; idx < right_logit_idx; idx += blockDim.x) {
    int logits_idx;
    if (in_col32) {
      int row_id = logits_token_idx_in_batch;
      int col_id = idx - logits_token_idx_in_batch * vocab_size;
      logits_idx = row_major2flat_col32(row_id, col_id,
                                        gridDim.x * logits_seq_len, vocab_size);
    } else {
      logits_idx = idx;
    }
    rough_top_kth_logit = fmaxf(
        rough_top_kth_logit,
        (float)(logits[logits_idx]) * dequant_scale +
            (float)__ldg(&logit_bias[idx - left_logit_idx + threadIdx.x]));
  }
  float max_logit = blockReduceMax(rough_top_kth_logit);
  rough_top_kth_logit = blockRoughTopK<float, k>(rough_top_kth_logit);
  if (threadIdx.x == 0) {
    s_topk_logit = rough_top_kth_logit;
    s_max_logit = max_logit;
  }
  __syncthreads();

  __shared__ int s_tid;

  if (k != 1) {
    /* step2 hold one logit per thread which larger than Kth logit and sample
     * from them */
    float topk_exp_sum, topk_exp = CUDA_FLOAT_INF_NEG;
    int topk_tid = vocab_size;
    // int test_num = 0;
    __shared__ float s_topk_exp_sum;
    for (int idx = left_logit_idx; idx < right_logit_idx; idx += blockDim.x) {
      int logits_idx;
      if (in_col32) {
        int row_id = logits_token_idx_in_batch;
        int col_id = idx - logits_token_idx_in_batch * vocab_size;
        logits_idx = row_major2flat_col32(
            row_id, col_id, gridDim.x * logits_seq_len, vocab_size);
      } else {
        logits_idx = idx;
      }
      float logit =
          (float)logits[logits_idx] * dequant_scale +
          (float)__ldg(&logit_bias[idx - left_logit_idx + threadIdx.x]);
      float logit_exp = expf(fmaxf(logit - s_max_logit, logit_thresh_min));
      // if (logit >= s_topk_logit) test_num++;
      if (logit >= s_topk_logit && logit_exp > topk_exp) {
        topk_exp = logit_exp;
        topk_tid = idx - left_logit_idx + threadIdx.x;
      }
    }

    // test_num = blockReduceSum(test_num);
    // __shared__ int s_test_num;
    // if (threadIdx.x == 0) {
    //   s_test_num = test_num;
    //   if (s_test_num != 1) printf("sample from top %d\n", s_test_num);
    //   // printf("sample from top %s", test_num);
    // }
    // __syncthreads();

    if (topk_tid == vocab_size) topk_exp = 0;
    topk_exp_sum = blockReduceSum(topk_exp);
    if (threadIdx.x == 0) {
      s_topk_exp_sum = topk_exp_sum;
    }
    __syncthreads();

    /* calculate cumulative probability */
    float topk_prob = topk_exp / s_topk_exp_sum;
    float prefix_sum_prob;
    typedef cub::BlockScan<float, 1024> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    BlockScan(temp_storage).InclusiveSum(topk_prob, prefix_sum_prob);

    __shared__ float random_x;
    if (threadIdx.x == 0) {
      random_x = curand_uniform(curandstate + blockIdx.x);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      s_tid = vocab_size;
    }
    __syncthreads();

    int threadID = threadIdx.x;
    __shared__ int s_threadID;
    __shared__ float s_max_prob;
    if (random_x > prefix_sum_prob) threadID = blockDim.x;
    threadID = blockReduceMin(threadID);
    float max_prob = blockReduceMax(topk_prob);
    if (threadIdx.x == 0) {
      s_threadID = threadID;
      s_max_prob = max_prob;
    }
    __syncthreads();
    if (threadIdx.x == s_threadID) {
      s_tid = topk_tid;
    }
    __syncthreads();

    if (s_tid == vocab_size && topk_prob == s_max_prob) {
      s_tid = topk_tid;
    }
    __syncthreads();
  } else {
    s_tid = vocab_size;
    for (int idx = left_logit_idx; idx < right_logit_idx; idx += blockDim.x) {
      int logits_idx;
      if (in_col32) {
        int row_id = logits_token_idx_in_batch;
        int col_id = idx - logits_token_idx_in_batch * vocab_size;
        logits_idx = row_major2flat_col32(
            row_id, col_id, gridDim.x * logits_seq_len, vocab_size);
      } else {
        logits_idx = idx;
      }
      float logit =
          (float)logits[logits_idx] * dequant_scale +
          (float)__ldg(&logit_bias[idx - left_logit_idx + threadIdx.x]);
      if (logit == s_max_logit) {
        s_tid = idx - left_logit_idx + threadIdx.x;
      }
    }
    __syncthreads();
  }

  /* if new sampled tid is not EOS, set unfinish TRUE */
  if (threadIdx.x == 0) {
    if (s_tid != eos_id) unfinished[0] = 1;
  }

  /* step3 write back new sampled ids */
  if (threadIdx.x == 0) {
    old_input_ids[last_token_idx_in_batch + 1] = s_tid;
  }
}

template <typename T>
void ker_topk_sample_i8I_launcher(
    int batch_size, int batch_seq_len, const int max_step, int logits_seq_len,
    int max_thread_per_block, cudaStream_t stream, const int8_t *logits,
    const T *logit_bias, int *old_input_ids, int *new_input_ids,
    const int vocab_size, const int k, int *unfinished,
    curandState *curandstate, int eos_id, float dequant_scale, bool in_col32) {
  if (k == 1)
    ker_topk_sample_i8I<T, 1><<<batch_size, max_thread_per_block, 0, stream>>>(
        logits, logit_bias, old_input_ids, new_input_ids, vocab_size, max_step,
        batch_seq_len, logits_seq_len, unfinished, curandstate, eos_id,
        dequant_scale, in_col32);
  else if (k == 2)
    ker_topk_sample_i8I<T, 2><<<batch_size, max_thread_per_block, 0, stream>>>(
        logits, logit_bias, old_input_ids, new_input_ids, vocab_size, max_step,
        batch_seq_len, logits_seq_len, unfinished, curandstate, eos_id,
        dequant_scale, in_col32);
  else if (k == 4)
    ker_topk_sample_i8I<T, 4><<<batch_size, max_thread_per_block, 0, stream>>>(
        logits, logit_bias, old_input_ids, new_input_ids, vocab_size, max_step,
        batch_seq_len, logits_seq_len, unfinished, curandstate, eos_id,
        dequant_scale, in_col32);
  else if (k == 8)
    ker_topk_sample_i8I<T, 8><<<batch_size, max_thread_per_block, 0, stream>>>(
        logits, logit_bias, old_input_ids, new_input_ids, vocab_size, max_step,
        batch_seq_len, logits_seq_len, unfinished, curandstate, eos_id,
        dequant_scale, in_col32);
  else if (k == 16)
    ker_topk_sample_i8I<T, 16><<<batch_size, max_thread_per_block, 0, stream>>>(
        logits, logit_bias, old_input_ids, new_input_ids, vocab_size, max_step,
        batch_seq_len, logits_seq_len, unfinished, curandstate, eos_id,
        dequant_scale, in_col32);
  else if (k == 32)
    ker_topk_sample_i8I<T, 32><<<batch_size, max_thread_per_block, 0, stream>>>(
        logits, logit_bias, old_input_ids, new_input_ids, vocab_size, max_step,
        batch_seq_len, logits_seq_len, unfinished, curandstate, eos_id,
        dequant_scale, in_col32);
  else {
    throw std::invalid_argument("topk argument should be in [1,2,4,8,16,32]");
  }
}

template void ker_topk_sample_i8I_launcher<float>(
    int batch_size, int batch_seq_len, const int max_step, int logits_seq_len,
    int max_thread_per_block, cudaStream_t stream, const int8_t *logits,
    const float *logit_bias, int *old_input_ids, int *new_input_idx,
    const int vocab_size, const int k, int *unfinished,
    curandState *curandstate, int eos_id, float dequant_scale, bool in_col32);

template void ker_topk_sample_i8I_launcher<__half>(
    int batch_size, int batch_seq_len, const int max_step, int logits_seq_len,
    int max_thread_per_block, cudaStream_t stream, const int8_t *logits,
    const __half *logit_bias, int *old_input_ids, int *new_input_idx,
    const int vocab_size, const int k, int *unfinished,
    curandState *curandstate, int eos_id, float dequant_scale, bool in_col32);

template <typename T>
__global__ void ker_topp_sample_i8I(const int8_t *logits, const T *logit_bias,
                                    int *old_input_ids, int *new_input_ids,
                                    const int vocab_size, const int max_step,
                                    const int batch_seq_len, int logits_seq_len,
                                    int *unfinished, float p,
                                    curandState *curandstate, int eos_id,
                                    float dequant_scale, bool in_col32) {
  int token_idx_in_batch = blockIdx.x * max_step + batch_seq_len - 1;

  /* add EOS to end if last token is EOS */
  if (batch_seq_len > 1 && old_input_ids[token_idx_in_batch] == eos_id) {
    if (threadIdx.x == 0) {
      old_input_ids[token_idx_in_batch + 1] = eos_id;
    }
    return;
  }
  int logits_token_idx_in_batch =
      blockIdx.x * logits_seq_len + logits_seq_len - 1;
  int left_logit_idx = logits_token_idx_in_batch * vocab_size + threadIdx.x;
  int right_logit_idx = (logits_token_idx_in_batch + 1) * vocab_size;

  /* step1. find max logit in each thread and sample from these probs with
   * nucleus sampling */
  __shared__ float s_max_logit;
  float max_logit = CUDA_FLOAT_INF_NEG;
  for (int idx = left_logit_idx; idx < right_logit_idx; idx += blockDim.x) {
    int logits_idx;
    if (in_col32) {
      int row_id = logits_token_idx_in_batch;
      int col_id = idx - logits_token_idx_in_batch * vocab_size;
      logits_idx = row_major2flat_col32(row_id, col_id,
                                        gridDim.x * logits_seq_len, vocab_size);
    } else {
      logits_idx = idx;
    }
    max_logit = fmaxf(max_logit, (float)logits[logits_idx] * dequant_scale) +
                (float)__ldg(&logit_bias[idx - left_logit_idx + threadIdx.x]);
  }
  float max_logit_array[1];
  max_logit_array[0] = max_logit;
  typedef cub::BlockRadixSort<float, 1024, 1> BlockRadixSort;
  __shared__ typename BlockRadixSort::TempStorage sort_temp_storage;
  BlockRadixSort(sort_temp_storage).SortDescending(max_logit_array);
  float presum_max_logit_exp;
  max_logit = max_logit_array[0];

  float block_max_logit = blockReduceMax(max_logit);
  if (threadIdx.x == 0) {
    s_max_logit = block_max_logit;
  }
  __syncthreads();

  float biased_logit_exp =
      expf(fmaxf(max_logit - s_max_logit, logit_thresh_min));

  typedef cub::BlockScan<float, 1024> BlockScan;
  __shared__ typename BlockScan::TempStorage presum_temp_storage;
  BlockScan(presum_temp_storage)
      .InclusiveSum(biased_logit_exp, presum_max_logit_exp);

  float topp_exp_threshold;
  if (threadIdx.x == blockDim.x - 1) {
    topp_exp_threshold = p * presum_max_logit_exp;
  }
  __shared__ float s_presum_logit_exp_threshold;
  if (presum_max_logit_exp > topp_exp_threshold) {
    presum_max_logit_exp = CUDA_FLOAT_INF_NEG;
  }
  float logit_exp_threshold = blockReduceMax(presum_max_logit_exp);
  if (threadIdx.x == 0) {
    s_presum_logit_exp_threshold = logit_exp_threshold;
  }
  __syncthreads();

  __shared__ float s_logit_threshold;
  if (presum_max_logit_exp == s_presum_logit_exp_threshold) {
    s_logit_threshold = max_logit;
  }
  __syncthreads();

  /* step2 hold one logit per thread which larger than Kth logit and sample
   * from them */
  float topk_exp_sum, topk_exp = CUDA_FLOAT_INF_NEG;
  int topk_tid = vocab_size;
  int test_num = 0;
  __shared__ float s_topk_exp_sum;
  for (int idx = left_logit_idx; idx < right_logit_idx; idx += blockDim.x) {
    int logits_idx;
    if (in_col32) {
      int row_id = logits_token_idx_in_batch;
      int col_id = idx - logits_token_idx_in_batch * vocab_size;
      logits_idx = row_major2flat_col32(row_id, col_id,
                                        gridDim.x * logits_seq_len, vocab_size);
    } else {
      logits_idx = idx;
    }
    float logit = (float)logits[logits_idx] * dequant_scale +
                  (float)__ldg(&logit_bias[idx - left_logit_idx + threadIdx.x]);
    float logit_exp = expf(fmaxf(logit - s_max_logit, logit_thresh_min));
    if (logit >= s_logit_threshold) test_num++;
    if (logit >= s_logit_threshold && logit_exp > topk_exp) {
      topk_exp = logit_exp;
      topk_tid = idx - left_logit_idx + threadIdx.x;
    }
  }

  test_num = blockReduceSum(test_num);

  if (topk_tid == vocab_size) topk_exp = 0;
  topk_exp_sum = blockReduceSum(topk_exp);
  if (threadIdx.x == 0) {
    s_topk_exp_sum = topk_exp_sum;
  }
  __syncthreads();

  /* calculate cumulative probability */
  float topk_prob = topk_exp / s_topk_exp_sum;
  float prefix_sum_prob;
  BlockScan(presum_temp_storage).InclusiveSum(topk_prob, prefix_sum_prob);

  __shared__ float random_x;
  if (threadIdx.x == 0) {
    random_x = curand_uniform(curandstate + blockIdx.x);
  }
  __syncthreads();

  __shared__ int s_tid;
  if (threadIdx.x == 0) {
    s_tid = vocab_size;
  }
  __syncthreads();

  int threadID = threadIdx.x;
  __shared__ int s_threadID;
  __shared__ float s_max_prob;
  if (random_x > prefix_sum_prob) threadID = blockDim.x;
  threadID = blockReduceMin(threadID);
  float max_prob = blockReduceMax(topk_prob);
  if (threadIdx.x == 0) {
    s_threadID = threadID;
    s_max_prob = max_prob;
  }
  __syncthreads();
  if (threadIdx.x == s_threadID) {
    s_tid = topk_tid;
  }
  __syncthreads();

  if (s_tid == vocab_size && topk_prob == s_max_prob) {
    s_tid = topk_tid;
  }
  __syncthreads();

  /* if new sampled tid is not EOS, set unfinish TRUE */
  if (threadIdx.x == 0) {
    if (s_tid != eos_id) unfinished[0] = 1;
  }

  /* step3 write back new sampled ids */
  if (threadIdx.x == 0) {
    old_input_ids[token_idx_in_batch + 1] = s_tid;
  }
}

template <typename T>
void ker_topp_sample_i8I_launcher(
    int batch_size, int batch_seq_len, const int max_step, int logits_seq_len,
    int max_thread_per_block, cudaStream_t stream, const int8_t *logits,
    const T *logit_bias, int *old_input_ids, int *new_input_ids,
    const int vocab_size, const float p, int *unfinished,
    curandState *curandstate, int eos_id, float dequant_scale, bool in_col32) {
  ker_topp_sample_i8I<T><<<batch_size, max_thread_per_block, 0, stream>>>(
      logits, logit_bias, old_input_ids, new_input_ids, vocab_size, max_step,
      batch_seq_len, logits_seq_len, unfinished, p, curandstate, eos_id,
      dequant_scale, in_col32);
}

template void ker_topp_sample_i8I_launcher<float>(
    int batch_size, int batch_seq_len, const int max_step, int logits_seq_len,
    int max_thread_per_block, cudaStream_t stream, const int8_t *logits,
    const float *logit_bias, int *old_input_ids, int *new_input_idx,
    const int vocab_size, const float p, int *unfinished,
    curandState *curandstate, int eos_id, float dequant_scale, bool in_col32);

template void ker_topp_sample_i8I_launcher<__half>(
    int batch_size, int batch_seq_len, const int max_step, int logits_seq_len,
    int max_thread_per_block, cudaStream_t stream, const int8_t *logits,
    const __half *logit_bias, int *old_input_ids, int *new_input_idx,
    const int vocab_size, const float p, int *unfinished,
    curandState *curandstate, int eos_id, float dequant_scale, bool in_col32);

}  // namespace cuda
}  // namespace lightseq
