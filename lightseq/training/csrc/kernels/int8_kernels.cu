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

/**
 * @brief element-wise activation function on device, like Relu, Gelu
 *
 * @tparam enum class ActivationType, kRelu, kGelu
 * @tparam input type
 * @param any shape of float and __half2
 * @return same shape and type with input
 */
template <ActivationType, typename T>
__forceinline__ __device__ T activation_kernel_int8(T x);

template <>
__device__ float activation_kernel_int8<ActivationType::kGelu, float>(float x) {
  float cdf =
      0.5f *
      (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
  return x * cdf;
}

template <>
__device__ __half2
activation_kernel_int8<ActivationType::kGelu, __half2>(__half2 val) {
  __half2 val_pow3 = __hmul2(val, __hmul2(val, val));
  float2 tmp_pow = __half22float2(val_pow3);
  float2 tmp = __half22float2(val);

  tmp.x =
      0.5f *
      (1.0f + tanhf((0.7978845608028654f * (tmp.x + 0.044715f * tmp_pow.x))));
  tmp.y =
      0.5f *
      (1.0f + tanhf((0.7978845608028654f * (tmp.y + 0.044715f * tmp_pow.y))));
  return __hmul2(val, __float22half2_rn(tmp));
}

template <>
__device__ float activation_kernel_int8<ActivationType::kRelu, float>(float x) {
  return fmaxf(x, 0);
}

template <>
__device__ __half2
activation_kernel_int8<ActivationType::kRelu, __half2>(__half2 x) {
  return __floats2half2_rn(fmaxf(0.f, __half2float(x.x)),
                           fmaxf(0.f, __half2float(x.y)));
}

/**
 * @brief fused bias, activation, and dropout at the end of first ffn
 *
 * @thread
 * gridDim.x = hidden_size / 8
 * blockDim.x = 8
 * blockDim.y = 1024 / 8 = 128
 *
 * @tparam act_type activation function, like kRelu, kGelu
 * @param total_count total elements
 * @param ratio drop ratio
 * @param out [batch_size, seq_len, hidden_size], float and __half
 * @param in [batch_size, seq_len, hidden_size], float and __half
 * @param mask [batch_size, seq_len, hidden_size], uint8 type
 * @param bias [hidden_size], ffn bias
 * @param seed seed to curand
 * @param hidden_size
 * @return void
 */
template <ActivationType act_type>
__global__ void ls_dropout_act_bias_kernel_int32I_int8O(
    const int total_count, const float ratio, int8_t *__restrict__ out,
    const int32_t *__restrict__ in, uint8_t *__restrict__ mask,
    const float *__restrict__ bias, const int seed, const int hidden_size,
    float in_scale, float in_clip_max, float out_scale, float out_clip_max) {
  const float scale = 1.f / (1.f - ratio);
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 4 >= total_count) return;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, i, 0, &state);
  uint8_t m[4];

  char4 *out4 = reinterpret_cast<char4 *>(out);
  const int4 *data4 = reinterpret_cast<const int4 *>(in);
  const float4 *bias4 = reinterpret_cast<const float4 *>(bias);
  uint32_t *mask4 = reinterpret_cast<uint32_t *>(mask);
  float4 rand = curand_uniform4(&state);

  m[0] = (uint8_t)(rand.x > ratio);
  m[1] = (uint8_t)(rand.y > ratio);
  m[2] = (uint8_t)(rand.z > ratio);
  m[3] = (uint8_t)(rand.w > ratio);

  int bias_i = i % (hidden_size >> 2);
  uint32_t *m4 = reinterpret_cast<uint32_t *>(m);
  mask4[i] = m4[0];
  const int4 input4 = data4[i];
  const float4 b4 = __ldg(&bias4[bias_i]);
  float4 output4;

  output4.x = activation_kernel_int8<act_type, float>(
                  float(input4.x) / in_scale * in_clip_max + b4.x) *
              scale * m[0];
  output4.y = activation_kernel_int8<act_type, float>(
                  float(input4.y) / in_scale * in_clip_max + b4.y) *
              scale * m[1];
  output4.z = activation_kernel_int8<act_type, float>(
                  float(input4.z) / in_scale * in_clip_max + b4.z) *
              scale * m[2];
  output4.w = activation_kernel_int8<act_type, float>(
                  float(input4.w) / in_scale * in_clip_max + b4.w) *
              scale * m[3];

  char4 out_i4;
  out_i4.x = float2int8(output4.x, out_scale / out_clip_max, out_clip_max);
  out_i4.y = float2int8(output4.y, out_scale / out_clip_max, out_clip_max);
  out_i4.z = float2int8(output4.z, out_scale / out_clip_max, out_clip_max);
  out_i4.w = float2int8(output4.w, out_scale / out_clip_max, out_clip_max);
  out4[i] = out_i4;
}

template <ActivationType act_type>
__global__ void ls_dropout_act_bias_kernel_int32I_int8O(
    const int total_count, const float ratio, int8_t *__restrict__ out,
    const int32_t *__restrict__ in, uint8_t *__restrict__ mask,
    const __half *__restrict__ bias, const int seed, const int hidden_size,
    float in_scale, float in_clip_max, float out_scale, float out_clip_max) {
  const float scale = 1.f / (1.f - ratio);

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 8 >= total_count) return;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, i, 0, &state);

  const long4 *vals_long4 = reinterpret_cast<const long4 *>(in);
  int64_t *outs_i8 = reinterpret_cast<int64_t *>(out);
  const float4 *bias4 = reinterpret_cast<const float4 *>(bias);
  uint64_t *mask8 = reinterpret_cast<uint64_t *>(mask);

  uint8_t m[8];
  float4 rand = curand_uniform4(&state);
  m[0] = (uint8_t)(rand.x > ratio);
  m[1] = (uint8_t)(rand.y > ratio);
  m[2] = (uint8_t)(rand.z > ratio);
  m[3] = (uint8_t)(rand.w > ratio);
  rand = curand_uniform4(&state);
  m[4] = (uint8_t)(rand.x > ratio);
  m[5] = (uint8_t)(rand.y > ratio);
  m[6] = (uint8_t)(rand.z > ratio);
  m[7] = (uint8_t)(rand.w > ratio);
  uint64_t *m8 = reinterpret_cast<uint64_t *>(m);
  mask8[i] = *m8;

  int bias_i = i % (hidden_size >> 3);
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
        scale * m[j] *
        activation_kernel_int8<act_type, float>(
            float(val1[j]) / in_scale * in_clip_max + __half2float(b_half[j]));
    out_i1[j] = float2int8(out_f, out_scale / out_clip_max, out_clip_max);
  }
  outs_i8[i] = out_i8;
}

template <>
void launch_ls_dropout_act_bias_int32I_int8O<ActivationType::kGelu, float>(
    int8_t *out, const int32_t *vals, uint8_t *mask, const float *bias,
    int total_count, int dim, float ratio, float in_scale, float in_clip_max,
    float out_scale, float out_clip_max, cudaStream_t stream) {
  int grid_dim = total_count >> 10;
  ls_dropout_act_bias_kernel_int32I_int8O<ActivationType::kGelu>
      <<<grid_dim + 1, 256, 0, stream>>>(
          total_count, ratio, out, vals, mask, bias,
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count(),
          dim, in_scale, in_clip_max, out_scale, out_clip_max);
}

template <>
void launch_ls_dropout_act_bias_int32I_int8O<ActivationType::kGelu, __half>(
    int8_t *out, const int32_t *vals, uint8_t *mask, const __half *bias,
    int total_count, int dim, float ratio, float in_scale, float in_clip_max,
    float out_scale, float out_clip_max, cudaStream_t stream) {
  int grid_dim = total_count >> 11;
  ls_dropout_act_bias_kernel_int32I_int8O<ActivationType::kGelu>
      <<<grid_dim + 1, 256, 0, stream>>>(
          total_count, ratio, out, vals, mask, bias,
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count(),
          dim, in_scale, in_clip_max, out_scale, out_clip_max);
}

template <>
void launch_ls_dropout_act_bias_int32I_int8O<ActivationType::kRelu, float>(
    int8_t *out, const int32_t *vals, uint8_t *mask, const float *bias,
    int total_count, int dim, float ratio, float in_scale, float in_clip_max,
    float out_scale, float out_clip_max, cudaStream_t stream) {
  int grid_dim = total_count >> 10;
  ls_dropout_act_bias_kernel_int32I_int8O<ActivationType::kRelu>
      <<<grid_dim + 1, 256, 0, stream>>>(
          total_count, ratio, out, vals, mask, bias,
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count(),
          dim, in_scale, in_clip_max, out_scale, out_clip_max);
}

template <>
void launch_ls_dropout_act_bias_int32I_int8O<ActivationType::kRelu, __half>(
    int8_t *out, const int32_t *vals, uint8_t *mask, const __half *bias,
    int total_count, int dim, float ratio, float in_scale, float in_clip_max,
    float out_scale, float out_clip_max, cudaStream_t stream) {
  int grid_dim = total_count >> 11;
  ls_dropout_act_bias_kernel_int32I_int8O<ActivationType::kRelu>
      <<<grid_dim + 1, 256, 0, stream>>>(
          total_count, ratio, out, vals, mask, bias,
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count(),
          dim, in_scale, in_clip_max, out_scale, out_clip_max);
}

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
__global__ void bias_add_transform_20314_int32I(T *output, const int32_t *input,
                                                const T *bias, int dim_3,
                                                int dim_4,
                                                float scale_div_clip_max);

template <>
__global__ void bias_add_transform_20314_int32I<float>(
    float *output, const int32_t *input, const float *bias, int dim_3,
    int dim_4, float scale_div_clip_max) {
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

  const int4 *qkv4 = reinterpret_cast<const int4 *>(input);
  const float4 *bias4 = reinterpret_cast<const float4 *>(bias);
  float4 *res4 = reinterpret_cast<float4 *>(output);
  int4 vqkv4;
  float4 vbias4;
  float4 vres4;

#pragma unroll
  for (std::size_t i = threadIdx.x; i < dim_34; i += blockDim.x) {
    vqkv4 = qkv4[src_offset + i];
    vbias4 = __ldg(bias4 + bias_offset + i);
    vres4.x = float(vqkv4.x) / scale_div_clip_max + vbias4.x;
    vres4.y = float(vqkv4.y) / scale_div_clip_max + vbias4.y;
    vres4.z = float(vqkv4.z) / scale_div_clip_max + vbias4.z;
    vres4.w = float(vqkv4.w) / scale_div_clip_max + vbias4.w;

    int id3 = i / dim_4;
    int id4 = i % dim_4;
    int cur_trg_offset = flat_3dim(id3, 0, id4, dim_1, dim_4);
    res4[trg_offset + cur_trg_offset] = vres4;
  }
}

template <>
__global__ void bias_add_transform_20314_int32I<__half>(
    __half *output, const int32_t *input, const __half *bias, int dim_3,
    int dim_4, float scale_div_clip_max) {
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

  const long4 *qkv4 = reinterpret_cast<const long4 *>(input);
  const float4 *bias4 = reinterpret_cast<const float4 *>(bias);
  float4 *res4 = reinterpret_cast<float4 *>(output);
  long4 vqkv4;
  float4 vbias4;
  float4 vres4;
  int32_t *vqkv1 = reinterpret_cast<int32_t *>(&vqkv4);
  __half2 *h2_bias = reinterpret_cast<__half2 *>(&vbias4);
  __half2 *h2_res = reinterpret_cast<__half2 *>(&vres4);
  float tmp1, tmp2;

#pragma unroll
  for (std::size_t i = threadIdx.x; i < dim_34; i += blockDim.x) {
    vqkv4 = qkv4[src_offset + i];
    vbias4 = __ldg(bias4 + bias_offset + i);
#pragma unroll
    for (std::size_t j = 0; j < 4; ++j) {
      tmp1 = float(vqkv1[j * 2]) / scale_div_clip_max;
      tmp2 = float(vqkv1[j * 2 + 1]) / scale_div_clip_max;
      h2_res[j] = __hadd2(__floats2half2_rn(tmp1, tmp2), h2_bias[j]);
    }

    int id3 = i / dim_4;
    int id4 = i % dim_4;
    int cur_trg_offset = flat_3dim(id3, 0, id4, dim_1, dim_4);
    res4[trg_offset + cur_trg_offset] = vres4;
  }
}

// [b, s, 3, h] -> [3, b, nh, s, ad]
template <>
void launch_bias_add_transform_20314_int32I<float>(
    float *output, const int32_t *input, const float *bias, int dim_0,
    int dim_1, int dim_2, int dim_3, int dim_4, float scale, float clip_max,
    cudaStream_t stream) {
  dim_4 >>= 2;

  dim3 grid_dim(dim_0, dim_1, dim_2);
  dim3 block_dim(min(dim_3 * dim_4, MAX_THREADS));

  bias_add_transform_20314_int32I<float><<<grid_dim, block_dim, 0, stream>>>(
      output, input, bias, dim_3, dim_4, scale / clip_max);
}

template <>
void launch_bias_add_transform_20314_int32I<__half>(
    __half *output, const int32_t *input, const __half *bias, int dim_0,
    int dim_1, int dim_2, int dim_3, int dim_4, float scale, float clip_max,
    cudaStream_t stream) {
  dim_4 >>= 3;

  dim3 grid_dim(dim_0, dim_1, dim_2);
  dim3 block_dim(min(dim_3 * dim_4, MAX_THREADS));

  bias_add_transform_20314_int32I<__half><<<grid_dim, block_dim, 0, stream>>>(
      output, input, bias, dim_3, dim_4, scale / clip_max);
}

/**
 * @brief fused bias, dropout, and residual at the end of Attention and FFN,
 * store dropped position in mask, it's not in-place
 *
 * @thread
 * gridDim.x = total_count / 1024
 * blockDim.x = 1024
 *
 * @param total_count total elements
 * @param ratio drop ratio
 * @param out [batch_size, seq_len, hidden_size], float and __half
 * @param in [batch_size, seq_len, hidden_size], float and __half
 * @param mask [batch_size, seq_len, hidden_size], uint8 type
 * @param bias [hidden_size], ffn bias
 * @param residual [batch_size, seq_len, hidden_size], float and __half
 * @param seed seed to curand
 * @param hidden_size hidden size
 * @return void
 */
__global__ void ls_dropout_res_bias_kernel_int32I(
    const int total_count, const float ratio, float *__restrict__ out,
    const int32_t *__restrict__ in, uint8_t *__restrict__ mask,
    const float *__restrict__ bias, const float *__restrict__ residual,
    const int seed, const int hidden_size, float quant_scale, float clip_max) {
  const float scale = 1.f / (1.f - ratio);
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 4 >= total_count) return;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, i, 0, &state);
  uint8_t m[4];

  float4 *out4 = reinterpret_cast<float4 *>(out);
  const int4 *data4 = reinterpret_cast<const int4 *>(in);
  const float4 *residual4 = reinterpret_cast<const float4 *>(residual);
  const float4 *bias4 = reinterpret_cast<const float4 *>(bias);
  uint32_t *mask4 = reinterpret_cast<uint32_t *>(mask);
  float4 rand = curand_uniform4(&state);

  m[0] = static_cast<uint8_t>(rand.x > ratio);
  m[1] = static_cast<uint8_t>(rand.y > ratio);
  m[2] = static_cast<uint8_t>(rand.z > ratio);
  m[3] = static_cast<uint8_t>(rand.w > ratio);

  int bias_i = i % (hidden_size >> 2);
  uint32_t *m4 = reinterpret_cast<uint32_t *>(m);
  mask4[i] = m4[0];
  const int4 input4 = data4[i];
  const float4 b4 = __ldg(&bias4[bias_i]);
  const float4 res4 = residual4[i];
  float4 output4;

  output4.x =
      (float(input4.x) / quant_scale * clip_max + b4.x) * scale * m[0] + res4.x;
  output4.y =
      (float(input4.y) / quant_scale * clip_max + b4.y) * scale * m[1] + res4.y;
  output4.z =
      (float(input4.z) / quant_scale * clip_max + b4.z) * scale * m[2] + res4.z;
  output4.w =
      (float(input4.w) / quant_scale * clip_max + b4.w) * scale * m[3] + res4.w;

  out4[i] = output4;
}

__global__ void ls_dropout_res_bias_kernel_int32I(
    const int total_count, const float ratio, __half *__restrict__ out,
    const int32_t *__restrict__ in, uint8_t *__restrict__ mask,
    const __half *__restrict__ bias, const __half *__restrict__ residual,
    const int seed, const int hidden_size, float quant_scale, float clip_max) {
  const __half scale = 1. / (1. - ratio);

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 8 >= total_count) return;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, i, 0, &state);

  const long4 *vals_long4 = reinterpret_cast<const long4 *>(in);
  float4 *outs_float4 = reinterpret_cast<float4 *>(out);
  const float4 *residual4 = reinterpret_cast<const float4 *>(residual);
  const float4 *bias4 = reinterpret_cast<const float4 *>(bias);
  uint64_t *mask8 = reinterpret_cast<uint64_t *>(mask);

  uint8_t m[8];
  float4 rand = curand_uniform4(&state);
  m[0] = static_cast<uint8_t>(rand.x > ratio);
  m[1] = static_cast<uint8_t>(rand.y > ratio);
  m[2] = static_cast<uint8_t>(rand.z > ratio);
  m[3] = static_cast<uint8_t>(rand.w > ratio);
  rand = curand_uniform4(&state);
  m[4] = static_cast<uint8_t>(rand.x > ratio);
  m[5] = static_cast<uint8_t>(rand.y > ratio);
  m[6] = static_cast<uint8_t>(rand.z > ratio);
  m[7] = static_cast<uint8_t>(rand.w > ratio);
  uint64_t *m8 = reinterpret_cast<uint64_t *>(m);
  mask8[i] = m8[0];

  int bias_i = i % (hidden_size >> 3);
  long4 val_long4 = vals_long4[i];
  const float4 b4 = __ldg(&bias4[bias_i]);
  const float4 res4 = residual4[i];
  float4 out_float4;

  int32_t *val_i1 = reinterpret_cast<int32_t *>(&val_long4);
  __half2 *out_half2 = reinterpret_cast<__half2 *>(&out_float4);
  const __half2 *b_half2 = reinterpret_cast<const __half2 *>(&b4);
  const __half2 *res_half2 = reinterpret_cast<const __half2 *>(&res4);
  __half2 scale_mask_1 =
      __halves2half2(scale * __float2half(m[0]), scale * __float2half(m[1]));
  __half2 scale_mask_2 =
      __halves2half2(scale * __float2half(m[2]), scale * __float2half(m[3]));
  __half2 scale_mask_3 =
      __halves2half2(scale * __float2half(m[4]), scale * __float2half(m[5]));
  __half2 scale_mask_4 =
      __halves2half2(scale * __float2half(m[6]), scale * __float2half(m[7]));
  out_half2[0] = __hfma2(
      __hadd2(__floats2half2_rn(float(val_i1[0]) / quant_scale * clip_max,
                                float(val_i1[1]) / quant_scale * clip_max),
              b_half2[0]),
      scale_mask_1, res_half2[0]);
  out_half2[1] = __hfma2(
      __hadd2(__floats2half2_rn(float(val_i1[2]) / quant_scale * clip_max,
                                float(val_i1[3]) / quant_scale * clip_max),
              b_half2[1]),
      scale_mask_2, res_half2[1]);
  out_half2[2] = __hfma2(
      __hadd2(__floats2half2_rn(float(val_i1[4]) / quant_scale * clip_max,
                                float(val_i1[5]) / quant_scale * clip_max),
              b_half2[2]),
      scale_mask_3, res_half2[2]);
  out_half2[3] = __hfma2(
      __hadd2(__floats2half2_rn(float(val_i1[6]) / quant_scale * clip_max,
                                float(val_i1[7]) / quant_scale * clip_max),
              b_half2[3]),
      scale_mask_4, res_half2[3]);
  outs_float4[i] = out_float4;
}

template <>
void launch_ls_dropout_res_bias_int32I<float>(
    float *out, const int32_t *vals, uint8_t *mask, const float *bias,
    const float *residual, int total_count, int dim, float ratio, float scale,
    float clip_max, cudaStream_t stream) {
  int grid_dim = total_count >> 12;
  ls_dropout_res_bias_kernel_int32I<<<grid_dim + 1, 1024, 0, stream>>>(
      total_count, ratio, out, vals, mask, bias, residual,
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count(),
      dim, scale, clip_max);
}

template <>
void launch_ls_dropout_res_bias_int32I<__half>(
    __half *out, const int32_t *vals, uint8_t *mask, const __half *bias,
    const __half *residual, int total_count, int dim, float ratio, float scale,
    float clip_max, cudaStream_t stream) {
  int grid_dim = total_count >> 13;
  ls_dropout_res_bias_kernel_int32I<<<grid_dim + 1, 1024, 0, stream>>>(
      total_count, ratio, out, vals, mask, bias, residual,
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count(),
      dim, scale, clip_max);
}

__global__ void transform4d_0213_int32I_int8O(
    int8_t *output, const int32_t *input, int batch_size, int seq_len,
    int trans_count, int nhead, int head_dim, int num_all, float in_scale,
    float in_clip_max, float out_scale, float out_clip_max) {
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

  const long4 *input8 = reinterpret_cast<const long4 *>(input);
  int64_t *res64 = reinterpret_cast<int64_t *>(output);
  long4 i8 = input8[offset];
  int32_t *i8s = reinterpret_cast<int32_t *>(&i8);
  int64_t res;
  int8_t *res8 = reinterpret_cast<int8_t *>(&res);
#pragma unroll
  for (std::size_t j = 0; j < 8; ++j) {
    res8[j] = float2int8(float(i8s[j]) / in_scale * in_clip_max,
                         out_scale / out_clip_max, out_clip_max);
  }
  res64[trg_offset] = res;
}

// [tc, b, nh, s, ad] -> [b, s, tc, nh, ad]
void launch_transform4d_0213_int32I_int8O(int8_t *output, const int32_t *input,
                                          int batch_size, int seq_len,
                                          int hidden_dim, int nhead,
                                          int trans_count, float in_scale,
                                          float in_clip_max, float out_scale,
                                          float out_clip_max,
                                          cudaStream_t stream) {
  hidden_dim >>= 3;
  int head_dim = hidden_dim / nhead;
  int num_all = batch_size * seq_len * trans_count * hidden_dim;
  int nblock = (num_all + MAX_THREADS - 1) / MAX_THREADS;

  transform4d_0213_int32I_int8O<<<nblock, MAX_THREADS, 0, stream>>>(
      output, input, batch_size, seq_len, trans_count, nhead, head_dim, num_all,
      in_scale, in_clip_max, out_scale, out_clip_max);
}

/**
 * @brief element-wise dropout, store dropped position in mask, it's not
 * in-place
 *
 * @thread
 * gridDim.x = total_count / 1024
 * blockDim.x = 1024
 *
 * @param total_count total elements
 * @param ratio drop ratio
 * @param out any size of float and __half
 * @param in same with out
 * @param mask uint8 type, same size with out
 * @param seed seed to curand
 * @return void
 */
__global__ void ls_dropout_kernel_int8O(
    const int total_count, const float ratio, int8_t *__restrict__ out,
    const float *__restrict__ in, uint8_t *__restrict__ mask, const int seed,
    float scale_div_clip_max, float clip_max) {
  const float scale = 1.f / (1.f - ratio);
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 4 >= total_count) return;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, i, 0, &state);
  uint8_t m[4];

  char4 *out4 = reinterpret_cast<char4 *>(out);
  const float4 *data4 = reinterpret_cast<const float4 *>(in);
  uint32_t *mask4 = reinterpret_cast<uint32_t *>(mask);
  float4 rand = curand_uniform4(&state);

  m[0] = (uint8_t)(rand.x > ratio);
  m[1] = (uint8_t)(rand.y > ratio);
  m[2] = (uint8_t)(rand.z > ratio);
  m[3] = (uint8_t)(rand.w > ratio);

  uint32_t *m4 = reinterpret_cast<uint32_t *>(m);
  mask4[i] = m4[0];

  float4 input4 = data4[i];
  char4 res4;
  res4.x = float2int8(input4.x * scale * m[0], scale_div_clip_max, clip_max);
  res4.y = float2int8(input4.y * scale * m[1], scale_div_clip_max, clip_max);
  res4.z = float2int8(input4.z * scale * m[2], scale_div_clip_max, clip_max);
  res4.w = float2int8(input4.w * scale * m[3], scale_div_clip_max, clip_max);
  out4[i] = res4;
}

__global__ void ls_dropout_kernel_int8O(
    const int total_count, const float ratio, int8_t *__restrict__ out,
    const __half *__restrict__ in, uint8_t *__restrict__ mask, const int seed,
    float scale_div_clip_max, float clip_max) {
  const float scale = 1.f / (1.f - ratio);

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 8 >= total_count) return;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, i, 0, &state);

  const float4 *vals_float4 = reinterpret_cast<const float4 *>(in);
  int64_t *outs_i64 = reinterpret_cast<int64_t *>(out);
  uint64_t *mask8 = reinterpret_cast<uint64_t *>(mask);

  uint8_t m[8];
  float4 rand = curand_uniform4(&state);
  m[0] = (uint8_t)(rand.x > ratio);
  m[1] = (uint8_t)(rand.y > ratio);
  m[2] = (uint8_t)(rand.z > ratio);
  m[3] = (uint8_t)(rand.w > ratio);
  rand = curand_uniform4(&state);
  m[4] = (uint8_t)(rand.x > ratio);
  m[5] = (uint8_t)(rand.y > ratio);
  m[6] = (uint8_t)(rand.z > ratio);
  m[7] = (uint8_t)(rand.w > ratio);
  uint64_t *m8 = reinterpret_cast<uint64_t *>(m);
  mask8[i] = *m8;

  float4 val_float4 = vals_float4[i];
  int64_t out_i64;
  __half2 *val_half2 = reinterpret_cast<__half2 *>(&val_float4);
  int8_t *out_i8 = reinterpret_cast<int8_t *>(&out_i64);
  __half2 scale_mask_1 = __floats2half2_rn(scale * m[0], scale * m[1]);
  __half2 scale_mask_2 = __floats2half2_rn(scale * m[2], scale * m[3]);
  __half2 scale_mask_3 = __floats2half2_rn(scale * m[4], scale * m[5]);
  __half2 scale_mask_4 = __floats2half2_rn(scale * m[6], scale * m[7]);

  __half2 tmp;
  tmp = __hmul2(val_half2[0], scale_mask_1);
  out_i8[0] = float2int8(__half2float(tmp.x), scale_div_clip_max, clip_max);
  out_i8[1] = float2int8(__half2float(tmp.y), scale_div_clip_max, clip_max);
  tmp = __hmul2(val_half2[1], scale_mask_2);
  out_i8[2] = float2int8(__half2float(tmp.x), scale_div_clip_max, clip_max);
  out_i8[3] = float2int8(__half2float(tmp.y), scale_div_clip_max, clip_max);
  tmp = __hmul2(val_half2[2], scale_mask_3);
  out_i8[4] = float2int8(__half2float(tmp.x), scale_div_clip_max, clip_max);
  out_i8[5] = float2int8(__half2float(tmp.y), scale_div_clip_max, clip_max);
  tmp = __hmul2(val_half2[3], scale_mask_4);
  out_i8[6] = float2int8(__half2float(tmp.x), scale_div_clip_max, clip_max);
  out_i8[7] = float2int8(__half2float(tmp.y), scale_div_clip_max, clip_max);
  outs_i64[i] = out_i64;
}

template <>
void launch_ls_dropout_int8O<float>(int8_t *out, const float *vals,
                                    uint8_t *mask, int total_count, float ratio,
                                    float scale, float clip_max,
                                    cudaStream_t stream, bool backward) {
  int grid_dim = total_count >> 12;

  ls_dropout_kernel_int8O<<<grid_dim + 1, 1024, 0, stream>>>(
      total_count, ratio, out, vals, mask,
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count(),
      scale / clip_max, clip_max);
}

template <>
void launch_ls_dropout_int8O<__half>(int8_t *out, const __half *vals,
                                     uint8_t *mask, int total_count,
                                     float ratio, float scale, float clip_max,
                                     cudaStream_t stream, bool backward) {
  int grid_dim = total_count >> 13;

  ls_dropout_kernel_int8O<<<grid_dim + 1, 1024, 0, stream>>>(
      total_count, ratio, out, vals, mask,
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count(),
      scale / clip_max, clip_max);
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
__global__ void transform4d_0213_int8O(int8_t *output, const float *input,
                                       int batch_size, int seq_len,
                                       int trans_count, int nhead, int head_dim,
                                       int num_all, float scale,
                                       float clip_max) {
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
  char4 *res32 = reinterpret_cast<char4 *>(output);
  float4 f4 = input4[offset];
  char4 res;
  res.x = float2int8(f4.x, scale / clip_max, clip_max);
  res.y = float2int8(f4.y, scale / clip_max, clip_max);
  res.z = float2int8(f4.z, scale / clip_max, clip_max);
  res.w = float2int8(f4.w, scale / clip_max, clip_max);
  res32[trg_offset] = res;
}

__global__ void transform4d_0213_int8O(int8_t *output, const __half *input,
                                       int batch_size, int seq_len,
                                       int trans_count, int nhead, int head_dim,
                                       int num_all, float scale,
                                       float clip_max) {
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
  int64_t *res64 = reinterpret_cast<int64_t *>(output);
  float4 f4 = input4[offset];
  __half *h8 = reinterpret_cast<__half *>(&f4);
  int64_t res;
  int8_t *res8 = reinterpret_cast<int8_t *>(&res);
#pragma unroll
  for (std::size_t j = 0; j < 8; ++j) {
    res8[j] = float2int8(__half2float(h8[j]), scale / clip_max, clip_max);
  }
  res64[trg_offset] = res;
}

// [tc, b, nh, s, ad] -> [b, s, tc, nh, ad]
template <>
void launch_transform4d_0213_int8O<float>(int8_t *output, const float *input,
                                          int batch_size, int seq_len,
                                          int hidden_dim, int nhead,
                                          int trans_count, float scale,
                                          float clip_max, cudaStream_t stream) {
  hidden_dim >>= 2;
  int head_dim = hidden_dim / nhead;
  int num_all = batch_size * seq_len * trans_count * hidden_dim;
  int nblock = (num_all + MAX_THREADS - 1) / MAX_THREADS;

  transform4d_0213_int8O<<<nblock, MAX_THREADS, 0, stream>>>(
      output, input, batch_size, seq_len, trans_count, nhead, head_dim, num_all,
      scale, clip_max);
}

template <>
void launch_transform4d_0213_int8O<__half>(int8_t *output, const __half *input,
                                           int batch_size, int seq_len,
                                           int hidden_dim, int nhead,
                                           int trans_count, float scale,
                                           float clip_max,
                                           cudaStream_t stream) {
  hidden_dim >>= 3;
  int head_dim = hidden_dim / nhead;
  int num_all = batch_size * seq_len * trans_count * hidden_dim;
  int nblock = (num_all + MAX_THREADS - 1) / MAX_THREADS;

  transform4d_0213_int8O<<<nblock, MAX_THREADS, 0, stream>>>(
      output, input, batch_size, seq_len, trans_count, nhead, head_dim, num_all,
      scale, clip_max);
}
