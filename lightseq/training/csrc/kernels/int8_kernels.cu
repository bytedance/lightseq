#include <chrono>
#include <ctime>

#include "block_reduce.h"
#include "int8_kernels.h"

const float LN_EPSILON = 1e-8f;

__forceinline__ __host__ __device__ int8_t float2int8(float x,
                                                      float scale_div_clipmax,
                                                      float clip_max) {
  x = x > clip_max ? clip_max : (x < -clip_max ? -clip_max : x);
  return int8_t(x * scale_div_clipmax);
}

template <typename T>
__global__ void quantize_tensor_kernel(const T *input, int8_t *output,
                                       int total_count, float scale,
                                       float clip_max);

template <>
__global__ void quantize_tensor_kernel<float>(const float *input,
                                              int8_t *output, int total_count,
                                              float scale_div_clipmax,
                                              float clip_max) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i * 4 >= total_count) return;

  const float4 *input4 = reinterpret_cast<const float4 *>(input);
  int32_t *output4 = reinterpret_cast<int32_t *>(output);
  float4 inp4 = input4[i];
  int32_t out4;
  int8_t *out1 = reinterpret_cast<int8_t *>(&out4);
  out1[0] = float2int8(inp4.x, scale_div_clipmax, clip_max);
  out1[1] = float2int8(inp4.y, scale_div_clipmax, clip_max);
  out1[2] = float2int8(inp4.z, scale_div_clipmax, clip_max);
  out1[3] = float2int8(inp4.w, scale_div_clipmax, clip_max);
  output4[i] = out4;
}

template <>
__global__ void quantize_tensor_kernel<__half>(const __half *input,
                                               int8_t *output, int total_count,
                                               float scale_div_clipmax,
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
    out1[j] = float2int8(__half2float(inp_h[j]), scale_div_clipmax, clip_max);
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
                                         float scale_div_clipmax,
                                         float clip_max);

template <>
__global__ void dequantize_tensor_kernel<float>(const int32_t *input,
                                                float *output, int total_count,
                                                float scale_div_clipmax,
                                                float clip_max) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i * 4 >= total_count) return;

  const int4 *input4 = reinterpret_cast<const int4 *>(input);
  float4 *output4 = reinterpret_cast<float4 *>(output);
  int4 inp4 = input4[i];
  float4 out4;
  out4.x = float(inp4.x) / scale_div_clipmax;
  out4.y = float(inp4.y) / scale_div_clipmax;
  out4.z = float(inp4.z) / scale_div_clipmax;
  out4.w = float(inp4.w) / scale_div_clipmax;
  output4[i] = out4;
}

template <>
__global__ void dequantize_tensor_kernel<__half>(const int32_t *input,
                                                 __half *output,
                                                 int total_count,
                                                 float scale_div_clipmax,
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
    out1[j] = __float2half(float(inp1[j]) / scale_div_clipmax);
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

void trans_weight(int8_t *input, int8_t *output, int m, int n,
                  cudaStream_t &stream) {
  cublasLtHandle_t handle;
  cublasLtCreate(&handle);
  cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;
  cublasLtMatrixLayout_t input_desc, output_desc;
  cublasLtMatrixTransformDesc_t transform_desc;
  cublasOperation_t opTrans = CUBLAS_OP_T;
  int ld_input = n, ld_output = 32 * m;
  float alpha = 1.0, beta = 0.0;
  CHECK_GPU_ERROR(
      cublasLtMatrixLayoutCreate(&input_desc, CUDA_R_8I, n, m, ld_input));
  CHECK_GPU_ERROR(
      cublasLtMatrixLayoutCreate(&output_desc, CUDA_R_8I, m, n, ld_output));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutSetAttribute(
      output_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32,
      sizeof(order_COL32)));

  CHECK_GPU_ERROR(
      cublasLtMatrixTransformDescCreate(&transform_desc, CUDA_R_32F));
  CHECK_GPU_ERROR(cublasLtMatrixTransformDescSetAttribute(
      transform_desc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTrans,
      sizeof(opTrans)));
  CHECK_GPU_ERROR(cublasLtMatrixTransform(handle, transform_desc, &alpha, input,
                                          input_desc, &beta, NULL, NULL, output,
                                          output_desc, stream));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(input_desc));
  CHECK_GPU_ERROR(cublasLtMatrixLayoutDestroy(output_desc));
  CHECK_GPU_ERROR(cublasLtMatrixTransformDescDestroy(transform_desc));
}

template <typename T>
void quant_trans_weight(const T *input, int8_t *output, int m, int n,
                        float scale, float clip_max) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int8_t *buffer = cuda_malloc<int8_t>(m * n);
  launch_quantize_tensor(input, buffer, m * n, scale, clip_max, stream);
  trans_weight(buffer, output, m, n, stream);
  cuda_free(buffer);
}

template void quant_trans_weight<float>(const float *input, int8_t *output,
                                        int m, int n, float scale,
                                        float clip_max);
template void quant_trans_weight<__half>(const __half *input, int8_t *output,
                                         int m, int n, float scale,
                                         float clip_max);

/**
@brief: ker_layer_norm
Standard layer normalization.
It will not only output the layer norm result,
  but also outputs variance.
  may also output means, depends on whether
  the means argument is nullptr

@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
ln_res: [batch_size* seq_len, hidden_size], ln result.
vars: [batch_size* seq_len], variance per token
means: [batch_size* seq_len], means per token, can be nullput
inp: [batch_size * seq_len, hidden_size], ln input.
scale: [hidden_size], ln scale
bias: [hidden_size], ln bias
*/
template <typename T>
__global__ void ker_layer_norm_int8O(int8_t *ln_res, T *vars, T *means,
                                     const T *inp, const T *scale,
                                     const T *bias, int hidden_size,
                                     float quant_scale, float clip_max) {
  // step 0. compute local sum
  float l_sum = 0;
  float l_square_sum = 0;
  const float4 *inp_f4 = (const float4 *)inp + blockIdx.x * hidden_size;
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    l_sum += val.x + val.y + val.z + val.w;
    l_square_sum +=
        val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
  }

  // step 1. compute reduce sum
  float mean_dim = float(hidden_size) * 4.f;
  float reduce_val[2] = {l_sum, l_square_sum};
  blockReduce<ReduceType::kSum, 2>(reduce_val);
  __shared__ float s_mean, s_var;
  if (threadIdx.x == 0) {
    s_mean = reduce_val[0] / mean_dim;
    if (means != nullptr) {
      means[blockIdx.x] = s_mean;
    }
    s_var = reduce_val[1] / mean_dim - s_mean * s_mean + LN_EPSILON;
    vars[blockIdx.x] = s_var;
    s_var = rsqrtf(s_var);
  }
  __syncthreads();

  // step 2. layer norm result
  char4 *output_i4 = (char4 *)ln_res + blockIdx.x * hidden_size;
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 vscale = __ldg((const float4 *)scale + idx);
    float4 vbias = __ldg((const float4 *)bias + idx);
    float4 val = inp_f4[idx];
    char4 val_i4;
    val.x = (val.x - s_mean) * s_var * vscale.x + vbias.x;
    val.y = (val.y - s_mean) * s_var * vscale.y + vbias.y;
    val.z = (val.z - s_mean) * s_var * vscale.z + vbias.z;
    val.w = (val.w - s_mean) * s_var * vscale.w + vbias.w;
    val_i4.x = float2int8(val.x, quant_scale / clip_max, clip_max);
    val_i4.y = float2int8(val.y, quant_scale / clip_max, clip_max);
    val_i4.z = float2int8(val.z, quant_scale / clip_max, clip_max);
    val_i4.w = float2int8(val.w, quant_scale / clip_max, clip_max);
    output_i4[idx] = val_i4;
  }
}

template <>
__global__ void ker_layer_norm_int8O<__half>(int8_t *ln_res, __half *vars,
                                             __half *means, const __half *inp,
                                             const __half *scale,
                                             const __half *bias,
                                             int hidden_size, float quant_scale,
                                             float clip_max) {
  // step 0. compute local sum
  float l_sum = 0;
  float l_square_sum = 0;
  const float4 *inp_f4 = (const float4 *)inp + blockIdx.x * hidden_size;
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val_f4 = inp_f4[idx];
    __half2 *val_h2 = (__half2 *)(&val_f4);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 val_f2 = __half22float2(val_h2[i]);
      l_sum += val_f2.x + val_f2.y;
      l_square_sum += val_f2.x * val_f2.x + val_f2.y * val_f2.y;
    }
  }

  // step 1. compute reduce sum
  float mean_dim = float(hidden_size) * 8.f;
  float reduce_val[2] = {l_sum, l_square_sum};
  blockReduce<ReduceType::kSum, 2>(reduce_val);
  __shared__ float s_mean, s_var;
  if (threadIdx.x == 0) {
    s_mean = reduce_val[0] / mean_dim;
    if (means != nullptr) {
      means[blockIdx.x] = s_mean;
    }
    s_var = reduce_val[1] / mean_dim - s_mean * s_mean + LN_EPSILON;
    vars[blockIdx.x] = s_var;
    s_var = rsqrtf(s_var);
  }
  __syncthreads();

  // step 2. layer norm result
  int64_t *output_i4 = (int64_t *)ln_res + blockIdx.x * hidden_size;
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    // load scale, bias, input
    float4 scale_f4 = __ldg((const float4 *)scale + idx);
    __half2 *scale_h2 = (__half2 *)(&scale_f4);
    float4 bias_f4 = __ldg((const float4 *)bias + idx);
    __half2 *bias_h2 = (__half2 *)(&bias_f4);
    float4 val_f4 = inp_f4[idx];
    __half2 *val_h2 = (__half2 *)(&val_f4);
    int64_t out8;
    int8_t *out1 = reinterpret_cast<int8_t *>(&out8);

#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 scale_f2 = __half22float2(scale_h2[i]);
      float2 bias_f2 = __half22float2(bias_h2[i]);
      float2 val_f2 = __half22float2(val_h2[i]);
      val_f2.x = (val_f2.x - s_mean) * s_var * scale_f2.x + bias_f2.x;
      val_f2.y = (val_f2.y - s_mean) * s_var * scale_f2.y + bias_f2.y;
      out1[i * 2] = float2int8(val_f2.x, quant_scale / clip_max, clip_max);
      out1[i * 2 + 1] = float2int8(val_f2.y, quant_scale / clip_max, clip_max);
    }
    output_i4[idx] = out8;
  }
}

template <>
void launch_layer_norm_int8O<float>(int8_t *ln_res, float *vars, float *means,
                                    const float *inp, const float *scale,
                                    const float *bias, int batch_size,
                                    int hidden_dim, float quant_scale,
                                    float clip_max, cudaStream_t stream) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm_int8O<float><<<grid_dim, block_dim, 0, stream>>>(
      ln_res, vars, means, inp, scale, bias, hidden_dim, quant_scale, clip_max);
}

template <>
void launch_layer_norm_int8O<__half>(int8_t *ln_res, __half *vars,
                                     __half *means, const __half *inp,
                                     const __half *scale, const __half *bias,
                                     int batch_size, int hidden_dim,
                                     float quant_scale, float clip_max,
                                     cudaStream_t stream) {
  if (hidden_dim % 8 != 0) {
    throw std::runtime_error("violate hidden_dim % 8 = 0");
  }
  hidden_dim >>= 3;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm_int8O<__half><<<grid_dim, block_dim, 0, stream>>>(
      ln_res, vars, means, inp, scale, bias, hidden_dim, quant_scale, clip_max);
}
