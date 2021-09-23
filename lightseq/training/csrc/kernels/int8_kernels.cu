#include <chrono>
#include <ctime>

#include "int8_kernels.h"

__host__ __device__ int8_t float2int(float x, float scale, float clip_max) {
  if (x > clip_max) x = clip_max;
  if (x < -clip_max) x = -clip_max;
  int8_t y = int8_t(x / clip_max * scale);
  return y;
}

template <typename T>
__global__ void quantize_tensor_kernel(const T *input, int8_t *output,
                                       int total_count, float scale,
                                       float clip_max);

template <>
__global__ void quantize_tensor_kernel<float>(const float *input,
                                              int8_t *output, int total_count,
                                              float scale, float clip_max) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i * 4 >= total_count) return;

  const float4 *input4 = reinterpret_cast<const float4 *>(input);
  int32_t *output4 = reinterpret_cast<int32_t *>(output);
  float4 inp4 = input4[i];
  int32_t out4;
  int8_t *out1 = reinterpret_cast<int8_t *>(&out4);
  out1[0] = float2int(inp4.x, scale, clip_max);
  out1[1] = float2int(inp4.y, scale, clip_max);
  out1[2] = float2int(inp4.z, scale, clip_max);
  out1[3] = float2int(inp4.w, scale, clip_max);
  output4[i] = out4;
}

template <>
__global__ void quantize_tensor_kernel<__half>(const __half *input,
                                               int8_t *output, int total_count,
                                               float scale, float clip_max) {
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
    out1[j] = float2int(__half2float(inp_h[j]), scale, clip_max);
  }
  output4[i] = out8;
}

template <>
void launch_quantize_tensor<float>(const float *input, int8_t *output,
                                   int total_count, float scale, float clip_max,
                                   cudaStream_t &stream) {
  int grid_dim = total_count >> 12;
  quantize_tensor_kernel<<<grid_dim + 1, 1024, 0, stream>>>(
      input, output, total_count, scale, clip_max);
}

template <>
void launch_quantize_tensor<__half>(const __half *input, int8_t *output,
                                    int total_count, float scale,
                                    float clip_max, cudaStream_t &stream) {
  int grid_dim = total_count >> 13;
  quantize_tensor_kernel<<<grid_dim + 1, 1024, 0, stream>>>(
      input, output, total_count, scale, clip_max);
}

template <typename T>
__global__ void dequantize_tensor_kernel(const int32_t *input, T *output,
                                         int total_count, float scale,
                                         float clip_max);

template <>
__global__ void dequantize_tensor_kernel<float>(const int32_t *input,
                                                float *output, int total_count,
                                                float scale, float clip_max) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i * 4 >= total_count) return;

  const int4 *input4 = reinterpret_cast<const int4 *>(input);
  float4 *output4 = reinterpret_cast<float4 *>(output);
  int4 inp4 = input4[i];
  float4 out4;
  out4.x = float(inp4.x) / scale * clip_max;
  out4.y = float(inp4.y) / scale * clip_max;
  out4.z = float(inp4.z) / scale * clip_max;
  out4.w = float(inp4.w) / scale * clip_max;
  output4[i] = out4;
}

template <>
__global__ void dequantize_tensor_kernel<__half>(const int32_t *input,
                                                 __half *output,
                                                 int total_count, float scale,
                                                 float clip_max) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i * 4 >= total_count) return;

  const int4 *input4 = reinterpret_cast<const int4 *>(input);
  float2 *output4 = reinterpret_cast<float2 *>(output);
  int4 inp4 = input4[i];
  float2 out4;
  int32_t *inp1 = reinterpret_cast<int32_t *>(&inp4);
  __half *out1 = reinterpret_cast<__half *>(&out4);
#pragma unroll
  for (uint j = 0; j < 4; ++j) {
    out1[j] = __float2half(float(inp1[j]) / scale * clip_max);
  }
  output4[i] = out4;
}

template <>
void launch_dequantize_tensor<float>(const int32_t *input, float *output,
                                     int total_count, float scale,
                                     float clip_max, cudaStream_t &stream) {
  int grid_dim = total_count >> 12;
  dequantize_tensor_kernel<<<grid_dim + 1, 1024, 0, stream>>>(
      input, output, total_count, scale, clip_max);
}

template <>
void launch_dequantize_tensor<__half>(const int32_t *input, __half *output,
                                      int total_count, float scale,
                                      float clip_max, cudaStream_t &stream) {
  int grid_dim = total_count >> 12;
  dequantize_tensor_kernel<<<grid_dim + 1, 1024, 0, stream>>>(
      input, output, total_count, scale, clip_max);
}
