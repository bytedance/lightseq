#include <chrono>
#include <ctime>

#include "int8_kernels.h"

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
