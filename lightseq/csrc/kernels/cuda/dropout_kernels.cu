#include <chrono>
#include <ctime>

#include "kernels.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;
namespace lightseq {
namespace cuda {

curandStatePhilox4_32_10_t *curandstate;

/**
 * @brief element-wise activation function on device, like Relu, Gelu
 *
 * @tparam enum class ActivationType, kRelu, kGelu
 * @tparam input type
 * @param any shape of float and __half2
 * @return same shape and type with input
 */
template <ActivationType, typename T>
__forceinline__ __device__ T activation_kernel(T x);

template <>
__device__ float activation_kernel<ActivationType::kGelu, float>(float x) {
  float cdf =
      0.5f *
      (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
  return x * cdf;
}

template <>
__device__ __half2
activation_kernel<ActivationType::kGelu, __half2>(__half2 val) {
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
__device__ float activation_kernel<ActivationType::kRelu, float>(float x) {
  return fmaxf(x, 0);
}

template <>
__device__ __half2
activation_kernel<ActivationType::kRelu, __half2>(__half2 x) {
  return __floats2half2_rn(fmaxf(0.f, __half2float(x.x)),
                           fmaxf(0.f, __half2float(x.y)));
}

/**
 * @brief element-wise activation backward function on device
 *
 * @tparam enum class ActivationType
 * @tparam input type
 * @param any shape of float and __half2
 * @return same shape of input
 */
template <ActivationType, typename T>
__forceinline__ __device__ T activation_bwd_kernel(T grad, T x);

template <>
__device__ float activation_bwd_kernel<ActivationType::kGelu, float>(float grad,
                                                                     float x) {
  const float sqrt_param = 0.79788456080286535587989211986876f;
  const float mul_param = 0.044715;

  float x2mul = x * x * mul_param;
  float tan_h = tanhf(sqrt_param * (x + x * x2mul));
  float dg1 = 0.5f * (1.0f + tan_h);
  float dg2 = x * 0.5f * sqrt_param * (1 - tan_h * tan_h);
  float dg3 = dg2 * 3 * x2mul;
  return grad * (dg1 + dg2 + dg3);
}

template <>
__device__ __half activation_bwd_kernel<ActivationType::kGelu, __half>(
    __half grad, __half x_half) {
  float x = __half2float(x_half);
  const float sqrt_param = 0.79788456080286535587989211986876f;
  const float mul_param = 0.044715;

  float x2mul = x * x * mul_param;
  float tan_h = tanhf(sqrt_param * (x + x * x2mul));
  float dg1 = 0.5f * (1.0f + tan_h);
  float dg2 = x * 0.5f * sqrt_param * (1 - tan_h * tan_h);
  float dg3 = dg2 * 3 * x2mul;
  return grad * __float2half(dg1 + dg2 + dg3);
}

template <>
__device__ float activation_bwd_kernel<ActivationType::kRelu, float>(float grad,
                                                                     float x) {
  return x > 0.f ? grad : 0.f;
}

template <>
__device__ __half
activation_bwd_kernel<ActivationType::kRelu, __half>(__half grad, __half x) {
  const __half half_zero = __float2half(0.f);
  return x > half_zero ? grad : half_zero;
}

template <>
__device__ __half2 activation_bwd_kernel<ActivationType::kRelu, __half2>(
    __half2 grad2, __half2 x_half2) {
  const __half half_zero = __float2half(0.f);
  return __floats2half2_rn(x_half2.x > half_zero ? grad2.x : half_zero,
                           x_half2.y > half_zero ? grad2.y : half_zero);
}

/**
 * @brief init curand states in global memory
 *
 * @thread grid_dim * block*dim to suuport any size of states
 * @param state persistant curand states
 * @param seed seed to init states
 * @return void
 */
__global__ void curand_init_kernel(curandStatePhilox4_32_10_t *state,
                                   int seed) {
  /* Each thread gets same seed, a different sequence
     number, no offset */
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, id, 0, &state[id]);
}

void launch_curand_init(int total_count, int dim, cudaStream_t stream) {
  cudaMalloc(&curandstate, total_count * sizeof(curandStatePhilox4_32_10_t));
  int grid_dim = total_count >> 9;
  curand_init_kernel<<<grid_dim, 512, 0, stream>>>(
      curandstate, std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count());
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
__global__ void ls_dropout_kernel(const int total_count, const float ratio,
                                  float *__restrict__ out,
                                  const float *__restrict__ in,
                                  uint8_t *__restrict__ mask, const int seed) {
  const float scale = 1.f / (1.f - ratio);
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 4 >= total_count) return;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, i, 0, &state);
  uint8_t m[4];

  float4 *out4 = reinterpret_cast<float4 *>(out);
  const float4 *data4 = reinterpret_cast<const float4 *>(in);
  uint32_t *mask4 = reinterpret_cast<uint32_t *>(mask);
  float4 rand = curand_uniform4(&state);

  m[0] = (uint8_t)(rand.x > ratio);
  m[1] = (uint8_t)(rand.y > ratio);
  m[2] = (uint8_t)(rand.z > ratio);
  m[3] = (uint8_t)(rand.w > ratio);

  uint32_t *m4 = reinterpret_cast<uint32_t *>(m);
  mask4[i] |= m4[0];

  float4 input4 = data4[i];
  float4 res4;
  res4.x = input4.x * scale * m[0];
  res4.y = input4.y * scale * m[1];
  res4.z = input4.z * scale * m[2];
  res4.w = input4.w * scale * m[3];
  out4[i] = res4;
}

__global__ void ls_dropout_kernel(const int total_count, const float ratio,
                                  __half *__restrict__ out,
                                  const __half *__restrict__ in,
                                  uint8_t *__restrict__ mask, const int seed) {
  const float scale = 1.f / (1.f - ratio);

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 8 >= total_count) return;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, i, 0, &state);

  const float4 *vals_float4 = reinterpret_cast<const float4 *>(in);
  float4 *outs_float4 = reinterpret_cast<float4 *>(out);
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
  mask8[i] |= *m8;

  float4 val_float4 = vals_float4[i];
  float4 out_float4;
  __half2 *val_half2 = reinterpret_cast<__half2 *>(&val_float4);
  __half2 *out_half2 = reinterpret_cast<__half2 *>(&out_float4);
  __half2 scale_mask_1 = __floats2half2_rn(scale * m[0], scale * m[1]);
  __half2 scale_mask_2 = __floats2half2_rn(scale * m[2], scale * m[3]);
  __half2 scale_mask_3 = __floats2half2_rn(scale * m[4], scale * m[5]);
  __half2 scale_mask_4 = __floats2half2_rn(scale * m[6], scale * m[7]);
  out_half2[0] = __hmul2(val_half2[0], scale_mask_1);
  out_half2[1] = __hmul2(val_half2[1], scale_mask_2);
  out_half2[2] = __hmul2(val_half2[2], scale_mask_3);
  out_half2[3] = __hmul2(val_half2[3], scale_mask_4);
  outs_float4[i] = out_float4;
}

/**
 * @brief element-wise dropout backward with dropout mask, it's
 * not in-place
 *
 * @thread
 * gridDim.x = total_count / 1024
 * blockDim.x = 1024
 *
 * @param total_count total elements
 * @param ratio drop ratio
 * @param in any size of float and __half
 * @param mask uint8 type, same size with in
 * @return void
 */
__global__ void ls_dropout_bwd_kernel(const int total_count, const float ratio,
                                      float *out, const float *in,
                                      const uint8_t *__restrict__ mask) {
  const float scale = 1.f / (1.f - ratio);
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 4 >= total_count) return;

  uint8_t m[4];

  float4 *out4 = reinterpret_cast<float4 *>(out);
  const float4 *in4 = reinterpret_cast<const float4 *>(in);
  const uint32_t *mask4 = reinterpret_cast<const uint32_t *>(mask);

  uint32_t *m4 = reinterpret_cast<uint32_t *>(m);
  m4[0] = mask4[i];

  float4 input4 = in4[i];
  float4 res4;
  res4.x = input4.x * scale * static_cast<float>(m[0] & 1);
  res4.y = input4.y * scale * static_cast<float>(m[1] & 1);
  res4.z = input4.z * scale * static_cast<float>(m[2] & 1);
  res4.w = input4.w * scale * static_cast<float>(m[3] & 1);
  out4[i] = res4;
}

__global__ void ls_dropout_bwd_kernel(const int total_count, const float ratio,
                                      __half *out, const __half *in,
                                      const uint8_t *__restrict__ mask) {
  const __half scale = 1.f / (1.f - ratio);

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 8 >= total_count) return;

  float4 *out4 = reinterpret_cast<float4 *>(out);
  const float4 *vals_float4 = reinterpret_cast<const float4 *>(in);
  const uint64_t *mask8 = reinterpret_cast<const uint64_t *>(mask);

  uint8_t m[8];
  uint64_t *m8 = reinterpret_cast<uint64_t *>(m);
  m8[0] = mask8[i];

  float4 val_float4 = vals_float4[i];
  float4 out_float4;
  __half2 *val_half2 = reinterpret_cast<__half2 *>(&val_float4);
  __half2 *out_half2 = reinterpret_cast<__half2 *>(&out_float4);
  __half2 scale_mask_1 = __halves2half2(scale * __float2half(m[0] & 1),
                                        scale * __float2half(m[1] & 1));
  __half2 scale_mask_2 = __halves2half2(scale * __float2half(m[2] & 1),
                                        scale * __float2half(m[3] & 1));
  __half2 scale_mask_3 = __halves2half2(scale * __float2half(m[4] & 1),
                                        scale * __float2half(m[5] & 1));
  __half2 scale_mask_4 = __halves2half2(scale * __float2half(m[6] & 1),
                                        scale * __float2half(m[7] & 1));
  out_half2[0] = __hmul2(val_half2[0], scale_mask_1);
  out_half2[1] = __hmul2(val_half2[1], scale_mask_2);
  out_half2[2] = __hmul2(val_half2[2], scale_mask_3);
  out_half2[3] = __hmul2(val_half2[3], scale_mask_4);
  out4[i] = out_float4;
}

template <>
void launch_ls_dropout<float>(float *out, const float *vals, uint8_t *mask,
                              int total_count, float ratio, cudaStream_t stream,
                              bool backward) {
  int grid_dim = total_count >> 12;
  if (!backward) {
    ls_dropout_kernel<<<grid_dim + 1, 1024, 0, stream>>>(
        total_count, ratio, out, vals, mask,
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
  } else {
    ls_dropout_bwd_kernel<<<grid_dim + 1, 1024, 0, stream>>>(total_count, ratio,
                                                             out, vals, mask);
  }
}

template <>
void launch_ls_dropout<__half>(__half *out, const __half *vals, uint8_t *mask,
                               int total_count, float ratio,
                               cudaStream_t stream, bool backward) {
  int grid_dim = total_count >> 13;
  if (!backward) {
    ls_dropout_kernel<<<grid_dim + 1, 1024, 0, stream>>>(
        total_count, ratio, out, vals, mask,
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
  } else {
    ls_dropout_bwd_kernel<<<grid_dim + 1, 1024, 0, stream>>>(total_count, ratio,
                                                             out, vals, mask);
  }
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
__global__ void ls_dropout_res_bias_kernel(
    const int total_count, const float ratio, float *__restrict__ out,
    const float *__restrict__ in, uint8_t *__restrict__ mask,
    const float *__restrict__ bias, const float *__restrict__ residual,
    const int seed, const int hidden_size) {
  const float scale = 1.f / (1.f - ratio);
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 4 >= total_count) return;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, i, 0, &state);
  uint8_t m[4];

  float4 *out4 = reinterpret_cast<float4 *>(out);
  const float4 *data4 = reinterpret_cast<const float4 *>(in);
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
  const float4 input4 = data4[i];
  const float4 b4 = __ldg(&bias4[bias_i]);
  const float4 res4 = residual4[i];
  float4 output4;

  output4.x = (input4.x + b4.x) * scale * m[0] + res4.x;
  output4.y = (input4.y + b4.y) * scale * m[1] + res4.y;
  output4.z = (input4.z + b4.z) * scale * m[2] + res4.z;
  output4.w = (input4.w + b4.w) * scale * m[3] + res4.w;

  out4[i] = output4;
}

__global__ void ls_dropout_res_bias_kernel(
    const int total_count, const float ratio, __half *__restrict__ out,
    const __half *__restrict__ in, uint8_t *__restrict__ mask,
    const __half *__restrict__ bias, const __half *__restrict__ residual,
    const int seed, const int hidden_size) {
  const __half scale = 1. / (1. - ratio);

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 8 >= total_count) return;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, i, 0, &state);

  const float4 *vals_float4 = reinterpret_cast<const float4 *>(in);
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
  float4 val_float4 = vals_float4[i];
  const float4 b4 = __ldg(&bias4[bias_i]);
  const float4 res4 = residual4[i];
  float4 out_float4;

  __half2 *val_half2 = reinterpret_cast<__half2 *>(&val_float4);
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
  out_half2[0] =
      __hfma2(__hadd2(val_half2[0], b_half2[0]), scale_mask_1, res_half2[0]);
  out_half2[1] =
      __hfma2(__hadd2(val_half2[1], b_half2[1]), scale_mask_2, res_half2[1]);
  out_half2[2] =
      __hfma2(__hadd2(val_half2[2], b_half2[2]), scale_mask_3, res_half2[2]);
  out_half2[3] =
      __hfma2(__hadd2(val_half2[3], b_half2[3]), scale_mask_4, res_half2[3]);
  outs_float4[i] = out_float4;
}

template <>
void launch_ls_dropout_res_bias<float>(float *out, const float *vals,
                                       uint8_t *mask, const float *bias,
                                       const float *residual, int total_count,
                                       int dim, float ratio,
                                       cudaStream_t stream) {
  int grid_dim = total_count >> 12;
  ls_dropout_res_bias_kernel<<<grid_dim + 1, 1024, 0, stream>>>(
      total_count, ratio, out, vals, mask, bias, residual,
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count(),
      dim);
}

template <>
void launch_ls_dropout_res_bias<__half>(__half *out, const __half *vals,
                                        uint8_t *mask, const __half *bias,
                                        const __half *residual, int total_count,
                                        int dim, float ratio,
                                        cudaStream_t stream) {
  int grid_dim = total_count >> 13;
  ls_dropout_res_bias_kernel<<<grid_dim + 1, 1024, 0, stream>>>(
      total_count, ratio, out, vals, mask, bias, residual,
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count(),
      dim);
}

/**
 * @brief fused bias and dropout backward at the end of Attention and FFN
 *
 * @thread
 * gridDim.x = hidden_size / 8
 * blockDim.x = 8
 * blockDim.y = 1024 / 8 = 128
 *
 * @param row_size batch_size * seq_len
 * @param ratio dropout ratio
 * @param in_grad [batch_size, seq_len, hidden_size], input grad
 * @param bias_grad [hidden_size], bias grad
 * @param out_grad [batch_size, seq_len, hidden_size], output grad
 * @param mask [batch_size, seq_len, hidden_size], dropout mask
 * @param hidden_size
 * @return void
 */
__global__ void ls_dropout_bias_bwd_kernel(
    const int row_size, const float ratio, float *__restrict__ in_grad,
    float *__restrict__ bias_grad, const float *__restrict__ out_grad,
    const uint8_t *__restrict__ mask, const int hidden_size) {
  const float scale = 1.f / (1.f - ratio);
  // every block generate 8 bias result
  __shared__ float tile[8][129];

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

  int col_idx = flat_2dim(blockIdx.x, threadIdx.x, 8);
  int stride = hidden_size * 128;
  float local_sum = 0;

  int idx = flat_2dim(threadIdx.y, col_idx, hidden_size);
  for (int r = threadIdx.y; r < row_size; r += 128) {
    float val = out_grad[idx];
    val *= scale * static_cast<float>(mask[idx] & 1);
    local_sum += val;
    in_grad[idx] = val;
    idx += stride;
  }

  tile[threadIdx.x][threadIdx.y] = local_sum;
  __syncthreads();

  float sum = 0;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int x = tid >> 7;
  int y = tid & (127);
  if (y < 32) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      sum += tile[x][y + i * 32];
    }
  }
  __syncthreads();

  for (int i = 1; i < 32; i <<= 1) sum += g.shfl_down(sum, i);

  if (y == 0) tile[0][x] = sum;
  __syncthreads();

  if (threadIdx.x < 8) {
    int pos = flat_2dim(blockIdx.x, threadIdx.x, 8);
    bias_grad[pos] = tile[0][threadIdx.x];
  }
}

__global__ void ls_dropout_bias_bwd_kernel(
    const int row_size, const float ratio, __half *__restrict__ in_grad,
    __half *__restrict__ bias_grad, const __half *__restrict__ out_grad,
    const uint8_t *__restrict__ mask, const int hidden_size) {
  const __half2 scale = __float2half2_rn(1.f / (1.f - ratio));
  __shared__ __half2 tile[8][129];

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

  __half2 *in_grad2 = reinterpret_cast<__half2 *>(in_grad);
  const __half2 *out_grad2 = reinterpret_cast<const __half2 *>(out_grad);
  __half2 *bias_grad2 = reinterpret_cast<__half2 *>(bias_grad);

  int col_idx = flat_2dim(blockIdx.x, threadIdx.x, 8);
  int stride = hidden_size * 128;
  __half2 local_sum = __float2half2_rn(0.f);

  int idx = flat_2dim(threadIdx.y, col_idx, hidden_size);
  for (int r = threadIdx.y; r < row_size; r += 128) {
    __half2 val = out_grad2[idx];
    __half2 m2 = __floats2half2_rn(mask[2 * idx] & 1, mask[2 * idx + 1] & 1);
    val *= scale * m2;
    local_sum += val;
    in_grad2[idx] = val;
    idx += stride;
  }

  tile[threadIdx.x][threadIdx.y] = local_sum;
  __syncthreads();

  __half2 sum = __float2half2_rn(0.f);
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int x = tid >> 7;
  int y = tid & (127);
  if (y < 32) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      sum += tile[x][y + i * 32];
    }
  }
  __syncthreads();

  for (int i = 1; i < WARP_SIZE; i <<= 1) sum += g.shfl_down(sum, i);

  if (y == 0) tile[0][x] = sum;
  __syncthreads();

  if (threadIdx.x < 8) {
    int pos = flat_2dim(blockIdx.x, threadIdx.x, 8);
    bias_grad2[pos] = tile[0][threadIdx.x];
  }
}

template <typename T>
void launch_ls_dropout_bias_bwd(T *in_grad, T *bias_grad, const T *out_grad,
                                const uint8_t *mask, int row_size, int dim,
                                float ratio, cudaStream_t stream) {
  dim3 grid_dim((dim - 1) / 8 + 1);
  dim3 block_dim(8, 128);
  ls_dropout_bias_bwd_kernel<<<grid_dim, block_dim, 0, stream>>>(
      row_size, ratio, in_grad, bias_grad, out_grad, mask, dim);
}

template <>
void launch_ls_dropout_bias_bwd(__half *in_grad, __half *bias_grad,
                                const __half *out_grad, const uint8_t *mask,
                                int row_size, int dim, float ratio,
                                cudaStream_t stream) {
  dim >>= 1;
  dim3 grid_dim((dim - 1) / 8 + 1);
  dim3 block_dim(8, 128);
  ls_dropout_bias_bwd_kernel<<<grid_dim, block_dim, 0, stream>>>(
      row_size, ratio, in_grad, bias_grad, out_grad, mask, dim);
}

template void launch_ls_dropout_bias_bwd(float *in_grad, float *bias_grad,
                                         const float *out_grad,
                                         const uint8_t *mask, int row_size,
                                         int dim, float ratio,
                                         cudaStream_t stream);

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
__global__ void ls_dropout_act_bias_kernel(
    const int total_count, const float ratio, float *__restrict__ out,
    const float *__restrict__ in, uint8_t *__restrict__ mask,
    const float *__restrict__ bias, const int seed, const int hidden_size) {
  const float scale = 1.f / (1.f - ratio);
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 4 >= total_count) return;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, i, 0, &state);
  uint8_t m[4];

  float4 *out4 = reinterpret_cast<float4 *>(out);
  const float4 *data4 = reinterpret_cast<const float4 *>(in);
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
  const float4 input4 = data4[i];
  const float4 b4 = __ldg(&bias4[bias_i]);
  float4 output4;

  output4.x =
      activation_kernel<act_type, float>(input4.x + b4.x) * scale * m[0];
  output4.y =
      activation_kernel<act_type, float>(input4.y + b4.y) * scale * m[1];
  output4.z =
      activation_kernel<act_type, float>(input4.z + b4.z) * scale * m[2];
  output4.w =
      activation_kernel<act_type, float>(input4.w + b4.w) * scale * m[3];

  out4[i] = output4;
}

template <ActivationType act_type>
__global__ void ls_dropout_act_bias_kernel(
    const int total_count, const float ratio, __half *__restrict__ out,
    const __half *__restrict__ in, uint8_t *__restrict__ mask,
    const __half *__restrict__ bias, const int seed, const int hidden_size) {
  const float scale = 1.f / (1.f - ratio);

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 8 >= total_count) return;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, i, 0, &state);

  const float4 *vals_float4 = reinterpret_cast<const float4 *>(in);
  float4 *outs_float4 = reinterpret_cast<float4 *>(out);
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
  float4 val_float4 = vals_float4[i];
  const float4 b4 = __ldg(&bias4[bias_i]);
  float4 out_float4;

  __half2 *val_half2 = reinterpret_cast<__half2 *>(&val_float4);
  __half2 *out_half2 = reinterpret_cast<__half2 *>(&out_float4);
  const __half2 *b_half2 = reinterpret_cast<const __half2 *>(&b4);

  __half2 scale_mask_1 = __floats2half2_rn(scale * m[0], scale * m[1]);
  __half2 scale_mask_2 = __floats2half2_rn(scale * m[2], scale * m[3]);
  __half2 scale_mask_3 = __floats2half2_rn(scale * m[4], scale * m[5]);
  __half2 scale_mask_4 = __floats2half2_rn(scale * m[6], scale * m[7]);
  out_half2[0] = __hmul2(
      activation_kernel<act_type, __half2>(__hadd2(val_half2[0], b_half2[0])),
      scale_mask_1);
  out_half2[1] = __hmul2(
      activation_kernel<act_type, __half2>(__hadd2(val_half2[1], b_half2[1])),
      scale_mask_2);
  out_half2[2] = __hmul2(
      activation_kernel<act_type, __half2>(__hadd2(val_half2[2], b_half2[2])),
      scale_mask_3);
  out_half2[3] = __hmul2(
      activation_kernel<act_type, __half2>(__hadd2(val_half2[3], b_half2[3])),
      scale_mask_4);
  outs_float4[i] = out_float4;
}

template <>
void launch_ls_dropout_act_bias<ActivationType::kGelu, float>(
    float *out, const float *vals, uint8_t *mask, const float *bias,
    int total_count, int dim, float ratio, cudaStream_t stream) {
  int grid_dim = total_count >> 10;
  ls_dropout_act_bias_kernel<ActivationType::kGelu>
      <<<grid_dim + 1, 256, 0, stream>>>(
          total_count, ratio, out, vals, mask, bias,
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count(),
          dim);
}

template <>
void launch_ls_dropout_act_bias<ActivationType::kGelu, __half>(
    __half *out, const __half *vals, uint8_t *mask, const __half *bias,
    int total_count, int dim, float ratio, cudaStream_t stream) {
  int grid_dim = total_count >> 11;
  ls_dropout_act_bias_kernel<ActivationType::kGelu>
      <<<grid_dim + 1, 256, 0, stream>>>(
          total_count, ratio, out, vals, mask, bias,
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count(),
          dim);
}

template <>
void launch_ls_dropout_act_bias<ActivationType::kRelu, float>(
    float *out, const float *vals, uint8_t *mask, const float *bias,
    int total_count, int dim, float ratio, cudaStream_t stream) {
  int grid_dim = total_count >> 10;
  ls_dropout_act_bias_kernel<ActivationType::kRelu>
      <<<grid_dim + 1, 256, 0, stream>>>(
          total_count, ratio, out, vals, mask, bias,
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count(),
          dim);
}

template <>
void launch_ls_dropout_act_bias<ActivationType::kRelu, __half>(
    __half *out, const __half *vals, uint8_t *mask, const __half *bias,
    int total_count, int dim, float ratio, cudaStream_t stream) {
  int grid_dim = total_count >> 11;
  ls_dropout_act_bias_kernel<ActivationType::kRelu>
      <<<grid_dim + 1, 256, 0, stream>>>(
          total_count, ratio, out, vals, mask, bias,
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count(),
          dim);
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
__global__ void ls_quant_dropout_act_bias_kernel(
    const int total_count, const float ratio, int8_t *qout, uint8_t *cmask_out,
    uint8_t *cmask_in, uint8_t *dropout_mask, const int8_t *qin,
    const float *bias, const float *cmax_out, const float *cmax_in,
    const int seed, const int hidden_size) {
  const float scale = 1.f / (1.f - ratio);
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 4 >= total_count) return;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, i, 0, &state);
  uint8_t m[4];

  float output_clip_max = cmax_out[0];
  float input_clip_max = cmax_in[0];

  int32_t *out4 = reinterpret_cast<int32_t *>(qout);
  const int32_t *qin4 = reinterpret_cast<const int32_t *>(qin);
  const float4 *bias4 = reinterpret_cast<const float4 *>(bias);
  uint32_t *dropout_mask4 = reinterpret_cast<uint32_t *>(dropout_mask);
  uint32_t *in_cmask4 = reinterpret_cast<uint32_t *>(cmask_in);
  float4 rand = curand_uniform4(&state);

  m[0] = (uint8_t)(rand.x > ratio);
  m[1] = (uint8_t)(rand.y > ratio);
  m[2] = (uint8_t)(rand.z > ratio);
  m[3] = (uint8_t)(rand.w > ratio);

  int bias_i = i % (hidden_size >> 2);
  uint32_t *m4 = reinterpret_cast<uint32_t *>(m);
  dropout_mask4[i] |= m4[0];
  int32_t qinput4 = qin4[i];
  int8_t *qinput = reinterpret_cast<int8_t *>(&qinput4);
  const float4 b4 = __ldg(&bias4[bias_i]);
  uint8_t in_cmask[4];
  int8_t out[4];

  out[0] = quantize(activation_kernel<act_type, float>(
                        dequantize(qinput[0], output_clip_max) + b4.x) *
                        scale * m[0],
                    input_clip_max, in_cmask[0], 2);
  out[1] = quantize(activation_kernel<act_type, float>(
                        dequantize(qinput[1], output_clip_max) + b4.y) *
                        scale * m[1],
                    input_clip_max, in_cmask[1], 2);
  out[2] = quantize(activation_kernel<act_type, float>(
                        dequantize(qinput[2], output_clip_max) + b4.z) *
                        scale * m[2],
                    input_clip_max, in_cmask[2], 2);
  out[3] = quantize(activation_kernel<act_type, float>(
                        dequantize(qinput[3], output_clip_max) + b4.w) *
                        scale * m[3],
                    input_clip_max, in_cmask[3], 2);

  in_cmask4[i] |= reinterpret_cast<uint32_t *>(in_cmask)[0];
  out4[i] = reinterpret_cast<int32_t *>(out)[0];
}

template <ActivationType act_type>
__global__ void ls_quant_dropout_act_bias_kernel(
    const int total_count, const float ratio, int8_t *qout, uint8_t *cmask_out,
    uint8_t *cmask_in, uint8_t *dropout_mask, const int8_t *qin,
    const __half *bias, const __half *cmax_out, const __half *cmax_in,
    const int seed, const int hidden_size) {
  const float scale = 1.f / (1.f - ratio);

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 8 >= total_count) return;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, i, 0, &state);

  const int64_t *qin8 = reinterpret_cast<const int64_t *>(qin);
  int64_t *qout8 = reinterpret_cast<int64_t *>(qout);
  const float4 *bias4 = reinterpret_cast<const float4 *>(bias);
  uint64_t *dropout_mask8 = reinterpret_cast<uint64_t *>(dropout_mask);
  uint64_t *in_cmask8 = reinterpret_cast<uint64_t *>(cmask_in);

  float output_clip_max = __half2float(cmax_out[0]);
  float input_clip_max = __half2float(cmax_in[0]);

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
  dropout_mask8[i] |= m8[0];

  int bias_i = i % (hidden_size >> 3);
  int64_t qinput8 = qin8[i];
  const float4 b4 = __ldg(&bias4[bias_i]);

  int8_t *qinput = reinterpret_cast<int8_t *>(&qinput8);
  int8_t out[8];
  const __half2 *b_half2 = reinterpret_cast<const __half2 *>(&b4);

  __half2 scale_mask[4];

  scale_mask[0] = __floats2half2_rn(scale * m[0], scale * m[1]);
  scale_mask[1] = __floats2half2_rn(scale * m[2], scale * m[3]);
  scale_mask[2] = __floats2half2_rn(scale * m[4], scale * m[5]);
  scale_mask[3] = __floats2half2_rn(scale * m[6], scale * m[7]);

  uint8_t in_cmask[8];

  __half2 temp;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    temp.x = __float2half(dequantize(qinput[j * 2], output_clip_max));
    temp.y = __float2half(dequantize(qinput[j * 2 + 1], output_clip_max));

    temp =
        __hmul2(activation_kernel<act_type, __half2>(__hadd2(temp, b_half2[j])),
                scale_mask[j]);

    out[j * 2] =
        quantize(__half2float(temp.x), input_clip_max, in_cmask[j * 2], 2);
    out[j * 2 + 1] =
        quantize(__half2float(temp.y), input_clip_max, in_cmask[j * 2 + 1], 2);
  }

  in_cmask8[i] |= reinterpret_cast<uint64_t *>(in_cmask)[0];
  qout8[i] = reinterpret_cast<int64_t *>(out)[0];
}

template <>
void launch_ls_quant_dropout_act_bias<ActivationType::kGelu, float>(
    int8_t *qout, uint8_t *cmask_out, uint8_t *cmask_in, uint8_t *dropout_mask,
    const int8_t *qinput, const float *bias, const float *cmax_out,
    const float *cmax_in, int total_count, int dim, float ratio,
    cudaStream_t stream) {
  int grid_dim = total_count >> 10;
  ls_quant_dropout_act_bias_kernel<ActivationType::kGelu>
      <<<grid_dim + 1, 256, 0, stream>>>(
          total_count, ratio, qout, cmask_out, cmask_in, dropout_mask, qinput,
          bias, cmax_out, cmax_in,
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count(),
          dim);
}

template <>
void launch_ls_quant_dropout_act_bias<ActivationType::kGelu, __half>(
    int8_t *qout, uint8_t *cmask_out, uint8_t *cmask_in, uint8_t *dropout_mask,
    const int8_t *qinput, const __half *bias, const __half *cmax_out,
    const __half *cmax_in, int total_count, int dim, float ratio,
    cudaStream_t stream) {
  int grid_dim = total_count >> 11;
  ls_quant_dropout_act_bias_kernel<ActivationType::kGelu>
      <<<grid_dim + 1, 256, 0, stream>>>(
          total_count, ratio, qout, cmask_out, cmask_in, dropout_mask, qinput,
          bias, cmax_out, cmax_in,
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count(),
          dim);
}

template <>
void launch_ls_quant_dropout_act_bias<ActivationType::kRelu, float>(
    int8_t *qout, uint8_t *cmask_out, uint8_t *cmask_in, uint8_t *dropout_mask,
    const int8_t *qinput, const float *bias, const float *cmax_out,
    const float *cmax_in, int total_count, int dim, float ratio,
    cudaStream_t stream) {
  int grid_dim = total_count >> 10;
  ls_quant_dropout_act_bias_kernel<ActivationType::kRelu>
      <<<grid_dim + 1, 256, 0, stream>>>(
          total_count, ratio, qout, cmask_out, cmask_in, dropout_mask, qinput,
          bias, cmax_out, cmax_in,
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count(),
          dim);
}

template <>
void launch_ls_quant_dropout_act_bias<ActivationType::kRelu, __half>(
    int8_t *qout, uint8_t *cmask_out, uint8_t *cmask_in, uint8_t *dropout_mask,
    const int8_t *qinput, const __half *bias, const __half *cmax_out,
    const __half *cmax_in, int total_count, int dim, float ratio,
    cudaStream_t stream) {
  int grid_dim = total_count >> 11;
  ls_quant_dropout_act_bias_kernel<ActivationType::kRelu>
      <<<grid_dim + 1, 256, 0, stream>>>(
          total_count, ratio, qout, cmask_out, cmask_in, dropout_mask, qinput,
          bias, cmax_out, cmax_in,
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count(),
          dim);
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
__global__ void ls_fakequant_dropout_act_bias_kernel(
    const int total_count, const float ratio, float *qout, uint8_t *cmask_out,
    uint8_t *cmask_in, uint8_t *dropout_mask, const int8_t *qin,
    const float *bias, const float *cmax_out, const float *cmax_in,
    const int seed, const int hidden_size, bool in_col32, bool symmetry) {
  const float scale = 1.f / (1.f - ratio);
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 4 >= total_count) return;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, i, 0, &state);
  uint8_t m[4];

  float output_clip_max = cmax_out[0];
  float input_clip_max = cmax_in[0];

  float4 *out4 = reinterpret_cast<float4 *>(qout);
  const int32_t *qin4 = reinterpret_cast<const int32_t *>(qin);
  const float4 *bias4 = reinterpret_cast<const float4 *>(bias);
  uint32_t *dropout_mask4 = reinterpret_cast<uint32_t *>(dropout_mask);
  uint32_t *in_cmask4 = reinterpret_cast<uint32_t *>(cmask_in);
  float4 rand = curand_uniform4(&state);

  m[0] = (uint8_t)(rand.x > ratio);
  m[1] = (uint8_t)(rand.y > ratio);
  m[2] = (uint8_t)(rand.z > ratio);
  m[3] = (uint8_t)(rand.w > ratio);

  int bias_i = i % (hidden_size >> 2);
  uint32_t *m4 = reinterpret_cast<uint32_t *>(m);
  dropout_mask4[i] |= m4[0];

  int input_index;
  if (in_col32) {
    int batch_tokens = total_count / hidden_size;
    int row_id = (i * 4) / hidden_size;
    int col_id = (i * 4) % hidden_size;
    input_index =
        row_major2flat_col32(row_id, col_id, batch_tokens, hidden_size) / 4;
  } else {
    input_index = i;
  }

  int32_t qinput4 = qin4[input_index];
  int8_t *qinput = reinterpret_cast<int8_t *>(&qinput4);
  const float4 b4 = __ldg(&bias4[bias_i]);
  uint8_t in_cmask[4];
  float4 out;

  out.x = fake_quantize(activation_kernel<act_type, float>(
                            dequantize(qinput[0], output_clip_max) + b4.x) *
                            scale * m[0],
                        input_clip_max, in_cmask[0], 2, symmetry);
  out.y = fake_quantize(activation_kernel<act_type, float>(
                            dequantize(qinput[1], output_clip_max) + b4.y) *
                            scale * m[1],
                        input_clip_max, in_cmask[1], 2, symmetry);
  out.z = fake_quantize(activation_kernel<act_type, float>(
                            dequantize(qinput[2], output_clip_max) + b4.z) *
                            scale * m[2],
                        input_clip_max, in_cmask[2], 2, symmetry);
  out.w = fake_quantize(activation_kernel<act_type, float>(
                            dequantize(qinput[3], output_clip_max) + b4.w) *
                            scale * m[3],
                        input_clip_max, in_cmask[3], 2, symmetry);

  in_cmask4[i] |= reinterpret_cast<uint32_t *>(in_cmask)[0];
  out4[i] = out;
}

template <ActivationType act_type>
__global__ void ls_fakequant_dropout_act_bias_kernel(
    const int total_count, const float ratio, __half *qout, uint8_t *cmask_out,
    uint8_t *cmask_in, uint8_t *dropout_mask, const int8_t *qin,
    const __half *bias, const __half *cmax_out, const __half *cmax_in,
    const int seed, const int hidden_size, bool in_col32, bool symmetry) {
  const float scale = 1.f / (1.f - ratio);

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 8 >= total_count) return;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, i, 0, &state);

  const int64_t *qin8 = reinterpret_cast<const int64_t *>(qin);
  float4 *qout8 = reinterpret_cast<float4 *>(qout);
  const float4 *bias4 = reinterpret_cast<const float4 *>(bias);
  uint64_t *dropout_mask8 = reinterpret_cast<uint64_t *>(dropout_mask);
  uint64_t *in_cmask8 = reinterpret_cast<uint64_t *>(cmask_in);

  float output_clip_max = __half2float(cmax_out[0]);
  float input_clip_max = __half2float(cmax_in[0]);

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
  dropout_mask8[i] |= m8[0];

  int bias_i = i % (hidden_size >> 3);
  int input_index;
  if (in_col32) {
    int batch_tokens = total_count / hidden_size;
    int row_id = (i * 8) / hidden_size;
    int col_id = (i * 8) % hidden_size;
    input_index =
        row_major2flat_col32(row_id, col_id, batch_tokens, hidden_size) / 8;
  } else {
    input_index = i;
  }
  int64_t qinput8 = qin8[input_index];
  const float4 b4 = __ldg(&bias4[bias_i]);

  int8_t *qinput = reinterpret_cast<int8_t *>(&qinput8);
  float4 out8;
  __half2 *out = reinterpret_cast<__half2 *>(&out8);
  const __half2 *b_half2 = reinterpret_cast<const __half2 *>(&b4);

  __half2 scale_mask[4];

  scale_mask[0] = __floats2half2_rn(scale * m[0], scale * m[1]);
  scale_mask[1] = __floats2half2_rn(scale * m[2], scale * m[3]);
  scale_mask[2] = __floats2half2_rn(scale * m[4], scale * m[5]);
  scale_mask[3] = __floats2half2_rn(scale * m[6], scale * m[7]);

  uint8_t in_cmask[8];

  __half2 temp;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    temp.x = __float2half(dequantize(qinput[j * 2], output_clip_max));
    temp.y = __float2half(dequantize(qinput[j * 2 + 1], output_clip_max));

    temp =
        __hmul2(activation_kernel<act_type, __half2>(__hadd2(temp, b_half2[j])),
                scale_mask[j]);

    out[j].x = __float2half(fake_quantize(__half2float(temp.x), input_clip_max,
                                          in_cmask[j * 2], 2, symmetry));
    out[j].y = __float2half(fake_quantize(__half2float(temp.y), input_clip_max,
                                          in_cmask[j * 2 + 1], 2, symmetry));
  }

  in_cmask8[i] |= reinterpret_cast<uint64_t *>(in_cmask)[0];
  qout8[i] = out8;
}

template <>
void launch_ls_fakequant_dropout_act_bias<ActivationType::kGelu, float>(
    float *out, uint8_t *cmask_out, uint8_t *cmask_in, uint8_t *dropout_mask,
    const int8_t *qinput, const float *bias, const float *cmax_out,
    const float *cmax_in, int total_count, int dim, float ratio,
    cudaStream_t stream, bool in_col32, bool symmetry) {
  int grid_dim = total_count >> 10;
  ls_fakequant_dropout_act_bias_kernel<ActivationType::kGelu>
      <<<grid_dim + 1, 256, 0, stream>>>(
          total_count, ratio, out, cmask_out, cmask_in, dropout_mask, qinput,
          bias, cmax_out, cmax_in,
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count(),
          dim, in_col32, symmetry);
}

template <>
void launch_ls_fakequant_dropout_act_bias<ActivationType::kGelu, __half>(
    __half *out, uint8_t *cmask_out, uint8_t *cmask_in, uint8_t *dropout_mask,
    const int8_t *qinput, const __half *bias, const __half *cmax_out,
    const __half *cmax_in, int total_count, int dim, float ratio,
    cudaStream_t stream, bool in_col32, bool symmetry) {
  int grid_dim = total_count >> 11;
  ls_fakequant_dropout_act_bias_kernel<ActivationType::kGelu>
      <<<grid_dim + 1, 256, 0, stream>>>(
          total_count, ratio, out, cmask_out, cmask_in, dropout_mask, qinput,
          bias, cmax_out, cmax_in,
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count(),
          dim, in_col32, symmetry);
}

template <>
void launch_ls_fakequant_dropout_act_bias<ActivationType::kRelu, float>(
    float *out, uint8_t *cmask_out, uint8_t *cmask_in, uint8_t *dropout_mask,
    const int8_t *qinput, const float *bias, const float *cmax_out,
    const float *cmax_in, int total_count, int dim, float ratio,
    cudaStream_t stream, bool in_col32, bool symmetry) {
  int grid_dim = total_count >> 10;
  ls_fakequant_dropout_act_bias_kernel<ActivationType::kRelu>
      <<<grid_dim + 1, 256, 0, stream>>>(
          total_count, ratio, out, cmask_out, cmask_in, dropout_mask, qinput,
          bias, cmax_out, cmax_in,
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count(),
          dim, in_col32, symmetry);
}

template <>
void launch_ls_fakequant_dropout_act_bias<ActivationType::kRelu, __half>(
    __half *out, uint8_t *cmask_out, uint8_t *cmask_in, uint8_t *dropout_mask,
    const int8_t *qinput, const __half *bias, const __half *cmax_out,
    const __half *cmax_in, int total_count, int dim, float ratio,
    cudaStream_t stream, bool in_col32, bool symmetry) {
  int grid_dim = total_count >> 11;
  ls_fakequant_dropout_act_bias_kernel<ActivationType::kRelu>
      <<<grid_dim + 1, 256, 0, stream>>>(
          total_count, ratio, out, cmask_out, cmask_in, dropout_mask, qinput,
          bias, cmax_out, cmax_in,
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count(),
          dim, in_col32, symmetry);
}

/**
 * @brief fused bias, activation, and dropout backward
 *
 * @thread
 * gridDim.x = total_count / 1024
 * blockDim.x = 1024
 *
 * @tparam act_type kRelu
 * @param row_size batch_size * seq_len
 * @param ratio dropout ratio
 * @param in_grad [batch_size, seq_len, hidden_size], input grad
 * @param bias_grad [hidden_size], bias grad
 * @param out_grad [batch_size, seq_len, hidden_size], output grad
 * @param mask [batch_size, seq_len, hidden_size], dropout mask
 * @param hidden_size
 * @return void
 */
template <ActivationType act_type, typename T>
__global__ void ls_dropout_act_bias_bwd_kernel(
    const int row_size, const float ratio, T *in_grad,
    T *__restrict__ bias_grad, const T *__restrict__ input,
    const T *__restrict__ bias, const T *out_grad,
    const uint8_t *__restrict__ mask, const int hidden_size) {
  const float scale = 1.f / (1.f - ratio);
  __shared__ float tile[WARP_SIZE][WARP_SIZE + 1];

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

  int col_idx = flat_2dim(blockIdx.x, threadIdx.x, WARP_SIZE);

  int stride = hidden_size * WARP_SIZE;
  float local_sum = 0;

  int idx = flat_2dim(threadIdx.y, col_idx, hidden_size);
  if (col_idx < hidden_size) {
    for (int r = threadIdx.y; r < row_size; r += WARP_SIZE) {
      float val = out_grad[idx];
      float in = input[idx];
      float b = bias[idx % hidden_size];
      val = activation_bwd_kernel<act_type, float>(
          val * scale * static_cast<float>(mask[idx]), in + b);
      local_sum += val;
      in_grad[idx] = val;
      idx += stride;
    }
  }

  tile[threadIdx.x][threadIdx.y] = local_sum;
  __syncthreads();
  float sum = tile[threadIdx.y][threadIdx.x];
  __syncthreads();

  for (int i = 1; i < WARP_SIZE; i <<= 1) sum += g.shfl_down(sum, i);

  if (threadIdx.x == 0) tile[0][threadIdx.y] = sum;
  __syncthreads();

  if (threadIdx.y == 0) {
    int pos = flat_2dim(blockIdx.x, threadIdx.x, WARP_SIZE);
    bias_grad[pos] = tile[0][threadIdx.x];
  }
}

// @brief fused bias, activation, and dropout backward
// It is deprecated for precision reason. Keep it for future optimization.
//
// template <ActivationType act_type>
// __global__ void ls_dropout_act_bias_bwd_kernel(
//     const int row_size, const float ratio, __half * in_grad,
//     __half *__restrict__ bias_grad, const __half *__restrict__ input, const
//     __half *__restrict__ bias, const __half * out_grad, const uint8_t
//     *__restrict__ mask, const int hidden_size) {
//   const __half2 scale = __float2half2_rn(1.f / (1.f - ratio));
//   __shared__ __half2 tile[WARP_SIZE][WARP_SIZE + 1];

//   cg::thread_block b = cg::this_thread_block();
//   cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

//   __half2 *in_grad2 = reinterpret_cast<__half2 *>(in_grad);
//   __half2 *bias_grad2 = reinterpret_cast<__half2 *>(bias_grad);
//   const __half2 *out_grad2 = reinterpret_cast<const __half2 *>(out_grad);
//   const __half2 *input2 = reinterpret_cast<const __half2 *>(input);
//   const __half2 *bias2 = reinterpret_cast<const __half2 *>(bias);

//   int col_idx = flat_2dim(blockIdx.x, threadIdx.x, WARP_SIZE);

//   int stride = hidden_size * WARP_SIZE;
//   __half2 local_sum = __float2half2_rn(0.f);

//   int idx = flat_2dim(threadIdx.y, col_idx, hidden_size);
//   if (col_idx < hidden_size) {
//     for (int r = threadIdx.y; r < row_size; r += WARP_SIZE) {
//       __half2 val = out_grad2[idx];
//       __half2 in2 = input2[idx];
//       __half2 b2 = bias2[idx % hidden_size ];
//       __half2 m2 = __floats2half2_rn(mask[2 * idx], mask[2 * idx + 1]);
//       val = activation_bwd_kernel<ActivationType::kRelu, __half2>(val * scale
//       *
//                                                                   m2,
//                                                                   in2+b2);
//       local_sum += val;
//       in_grad2[idx] = val;
//       idx += stride;
//     }
//   }

//   tile[threadIdx.x][threadIdx.y] = local_sum;
//   __syncthreads();
//   __half2 sum = tile[threadIdx.y][threadIdx.x];
//   __syncthreads();

//   for (int i = 1; i < WARP_SIZE; i <<= 1) sum += g.shfl_down(sum, i);

//   if (threadIdx.x == 0) tile[0][threadIdx.y] = sum;
//   __syncthreads();

//   if (threadIdx.y == 0) {
//     int pos = flat_2dim(blockIdx.x, threadIdx.x, WARP_SIZE);
//     bias_grad2[pos] = tile[0][threadIdx.x];
//   }
// }

template <ActivationType act_type, typename T>
void launch_ls_dropout_act_bias_bwd(T *in_grad, T *bias_grad, const T *input,
                                    const T *bias, const T *out_grad,
                                    const uint8_t *mask, int row_size, int dim,
                                    float ratio, cudaStream_t stream) {
  dim3 grid_dim((dim - 1) / WARP_SIZE + 1);
  dim3 block_dim(WARP_SIZE, WARP_SIZE);
  ls_dropout_act_bias_bwd_kernel<act_type><<<grid_dim, block_dim, 0, stream>>>(
      row_size, ratio, in_grad, bias_grad, input, bias, out_grad, mask, dim);
}

// template <>
// void launch_ls_dropout_act_bias_bwd<ActivationType::kRelu, __half>(
//     __half *in_grad, __half *bias_grad,const __half *input, const __half
//     *bias, const __half *out_grad, const uint8_t *mask, int row_size, int
//     dim, float ratio, cudaStream_t stream) {
//   dim >>= 1;
//   dim3 grid_dim((dim - 1) / WARP_SIZE + 1);
//   dim3 block_dim(WARP_SIZE, WARP_SIZE);
//   ls_dropout_act_bias_bwd_kernel<ActivationType::kRelu>
//       <<<grid_dim, block_dim, 0, stream>>>(row_size, ratio, in_grad,
//       bias_grad,
//                                            input, bias,out_grad, mask, dim);
// }

template void launch_ls_dropout_act_bias_bwd<ActivationType::kRelu, float>(
    float *in_grad, float *bias_grad, const float *input, const float *bias,
    const float *out_grad, const uint8_t *mask, int row_size, int dim,
    float ratio, cudaStream_t stream);

template void launch_ls_dropout_act_bias_bwd<ActivationType::kRelu, __half>(
    __half *in_grad, __half *bias_grad, const __half *input, const __half *bias,
    const __half *out_grad, const uint8_t *mask, int row_size, int dim,
    float ratio, cudaStream_t stream);

template void launch_ls_dropout_act_bias_bwd<ActivationType::kGelu, float>(
    float *in_grad, float *bias_grad, const float *input, const float *bias,
    const float *out_grad, const uint8_t *mask, int row_size, int dim,
    float ratio, cudaStream_t stream);

template void launch_ls_dropout_act_bias_bwd<ActivationType::kGelu, __half>(
    __half *in_grad, __half *bias_grad, const __half *input, const __half *bias,
    const __half *out_grad, const uint8_t *mask, int row_size, int dim,
    float ratio, cudaStream_t stream);

/**
 * @brief fused bias, activation, and dropout backward
 *
 * @thread
 * gridDim.x = total_count / 1024
 * blockDim.x = 1024
 *
 * @tparam act_type kRelu
 * @param row_size batch_size * seq_len
 * @param ratio dropout ratio
 * @param in_grad [batch_size, seq_len, hidden_size], input grad
 * @param bias_grad [hidden_size], bias grad
 * @param out_grad [batch_size, seq_len, hidden_size], output grad
 * @param mask [batch_size, seq_len, hidden_size], dropout mask
 * @param hidden_size
 * @return void
 */
template <ActivationType act_type, typename T>
__global__ void ls_quant_dropout_act_bias_bwd_kernel(
    T *in_grad, T *bias_grad, T *cmax_in_grad, T *cmax_out_grad,
    const int8_t *input, const T *cmax_in, const uint8_t *cmask_in,
    const uint8_t *cmask_out, const T *bias, const T *out_grad,
    const uint8_t *dropout_mask, int row_size, float ratio, int hidden_size,
    bool in_col32) {
  const float scale = 1.f / (1.f - ratio);
  __shared__ float tile[WARP_SIZE][WARP_SIZE + 1];

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

  int col_idx = flat_2dim(blockIdx.x, threadIdx.x, WARP_SIZE);

  int stride = hidden_size * WARP_SIZE;
  float thread_grad_bias = 0;

  float cmax_in_val = cmax_in[0];

  int idx = flat_2dim(threadIdx.y, col_idx, hidden_size);

  // float thread_cmax_out_grad = 0;
  float thread_cmax_in_grad = 0;
  float thread_in_grad = 0;
  float temp_cmax_in_grad = 0;
  // float temp_cmax_out_grad = 0;

  int input_index;

  if (col_idx < hidden_size) {
    for (int r = threadIdx.y; r < row_size; r += WARP_SIZE) {
      float val = out_grad[idx];
      // clip_bwd(thread_in_grad, temp_cmax_out_grad, float{out_grad[idx]},
      //          cmask_out[idx], 2);
      // thread_cmax_out_grad += temp_cmax_out_grad;
      if (in_col32) {
        int row_id = idx / hidden_size;
        int col_id = idx % hidden_size;
        input_index =
            row_major2flat_col32(row_id, col_id, row_size, hidden_size);
      } else {
        input_index = idx;
      }

      float in = dequantize(input[input_index], cmax_in_val);
      float b = bias[idx % hidden_size];
      uint8_t mask = dropout_mask[idx];
      thread_in_grad = activation_bwd_kernel<act_type, float>(
          val * scale * static_cast<float>(mask & 1), in + b);
      thread_grad_bias += thread_in_grad;

      clip_bwd(thread_in_grad, temp_cmax_in_grad, thread_in_grad, mask, 6);
      in_grad[idx] = thread_in_grad;
      thread_cmax_in_grad += temp_cmax_in_grad;
      idx += stride;
    }
  }
  __shared__ float block_cmax_in_grad;
  // __shared__ float block_cmax_out_grad;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    block_cmax_in_grad = 0;
    // block_cmax_out_grad = 0;
  }

  tile[threadIdx.x][threadIdx.y] = thread_grad_bias;
  __syncthreads();
  // if (thread_cmax_out_grad != 0) {
  //   atomicAdd(&block_cmax_out_grad, thread_cmax_out_grad);
  // }
  if (thread_cmax_in_grad != 0) {
    atomicAdd(&block_cmax_in_grad, thread_cmax_in_grad);
  }

  float sum = tile[threadIdx.y][threadIdx.x];

  __syncthreads();

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    if (block_cmax_in_grad != 0) {
      atomicAdd(&cmax_in_grad[0], block_cmax_in_grad);
    }
    // if (block_cmax_out_grad != 0) {
    //   atomicAdd(&cmax_out_grad[0], block_cmax_out_grad);
    // }
  }
  for (int i = 1; i < WARP_SIZE; i <<= 1) sum += g.shfl_down(sum, i);

  if (threadIdx.x == 0) tile[0][threadIdx.y] = sum;
  __syncthreads();

  if (threadIdx.y == 0) {
    int pos = flat_2dim(blockIdx.x, threadIdx.x, WARP_SIZE);
    bias_grad[pos] = tile[0][threadIdx.x];
  }
}

template <ActivationType act_type, typename T>
void launch_ls_quant_dropout_act_bias_bwd(
    T *in_grad, T *bias_grad, T *cmax_in_grad, T *cmax_out_grad,
    const int8_t *input, const T *cmax_in, const uint8_t *cmask_in,
    const uint8_t *cmask_out, const T *bias, const T *out_grad,
    const uint8_t *dropout_mask, int row_size, int dim, float ratio,
    cudaStream_t stream, bool in_col32) {
  zero_grad<<<1, 1>>>(cmax_in_grad);
  zero_grad<<<1, 1>>>(cmax_out_grad);
  dim3 grid_dim((dim - 1) / WARP_SIZE + 1);
  dim3 block_dim(WARP_SIZE, WARP_SIZE);
  ls_quant_dropout_act_bias_bwd_kernel<act_type>
      <<<grid_dim, block_dim, 0, stream>>>(
          in_grad, bias_grad, cmax_in_grad, cmax_out_grad, input, cmax_in,
          cmask_in, cmask_out, bias, out_grad, dropout_mask, row_size, ratio,
          dim, in_col32);
}

template void
launch_ls_quant_dropout_act_bias_bwd<ActivationType::kRelu, float>(
    float *in_grad, float *bias_grad, float *cmax_in_grad, float *cmax_out_grad,
    const int8_t *input, const float *cmax_in, const uint8_t *cmask_in,
    const uint8_t *cmask_out, const float *bias, const float *out_grad,
    const uint8_t *dropout_mask, int row_size, int dim, float ratio,
    cudaStream_t stream, bool in_col32);

template void
launch_ls_quant_dropout_act_bias_bwd<ActivationType::kRelu, __half>(
    __half *in_grad, __half *bias_grad, __half *cmax_in_grad,
    __half *cmax_out_grad, const int8_t *input, const __half *cmax_in,
    const uint8_t *cmask_in, const uint8_t *cmask_out, const __half *bias,
    const __half *out_grad, const uint8_t *dropout_mask, int row_size, int dim,
    float ratio, cudaStream_t stream, bool in_col32);

template void
launch_ls_quant_dropout_act_bias_bwd<ActivationType::kGelu, float>(
    float *in_grad, float *bias_grad, float *cmax_in_grad, float *cmax_out_grad,
    const int8_t *input, const float *cmax_in, const uint8_t *cmask_in,
    const uint8_t *cmask_out, const float *bias, const float *out_grad,
    const uint8_t *dropout_mask, int row_size, int dim, float ratio,
    cudaStream_t stream, bool in_col32);

template void
launch_ls_quant_dropout_act_bias_bwd<ActivationType::kGelu, __half>(
    __half *in_grad, __half *bias_grad, __half *cmax_in_grad,
    __half *cmax_out_grad, const int8_t *input, const __half *cmax_in,
    const uint8_t *cmask_in, const uint8_t *cmask_out, const __half *bias,
    const __half *out_grad, const uint8_t *dropout_mask, int row_size, int dim,
    float ratio, cudaStream_t stream, bool in_col32);

/**
 * @brief fused bias, activation, and dropout backward, with float input
 *
 * @thread
 * gridDim.x = total_count / 1024
 * blockDim.x = 1024
 *
 * @tparam act_type kRelu
 * @param row_size batch_size * seq_len
 * @param ratio dropout ratio
 * @param in_grad [batch_size, seq_len, hidden_size], input grad
 * @param bias_grad [hidden_size], bias grad
 * @param out_grad [batch_size, seq_len, hidden_size], output grad
 * @param mask [batch_size, seq_len, hidden_size], dropout mask
 * @param hidden_size
 * @return void
 */
template <ActivationType act_type, typename T>
__global__ void ls_quant_dropout_act_bias_bwd_kernel(
    T *in_grad, T *bias_grad, T *cmax_in_grad, T *cmax_out_grad, const T *input,
    const T *cmax_in, const uint8_t *cmask_in, const uint8_t *cmask_out,
    const T *bias, const T *out_grad, const uint8_t *dropout_mask, int row_size,
    float ratio, int hidden_size) {
  const float scale = 1.f / (1.f - ratio);
  __shared__ float tile[WARP_SIZE][WARP_SIZE + 1];

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

  int col_idx = flat_2dim(blockIdx.x, threadIdx.x, WARP_SIZE);

  int stride = hidden_size * WARP_SIZE;
  float thread_grad_bias = 0;

  float cmax_in_val = cmax_in[0];

  int idx = flat_2dim(threadIdx.y, col_idx, hidden_size);

  float thread_cmax_out_grad = 0;
  float thread_cmax_in_grad = 0;
  float thread_in_grad = 0;
  float temp_cmax_in_grad = 0;
  float temp_cmax_out_grad = 0;
  if (col_idx < hidden_size) {
    for (int r = threadIdx.y; r < row_size; r += WARP_SIZE) {
      // float val = out_grad[idx];
      clip_bwd(thread_in_grad, temp_cmax_out_grad, float{out_grad[idx]},
               cmask_out[idx], 2);
      thread_cmax_out_grad += temp_cmax_out_grad;

      float in = input[idx];
      float b = bias[idx % hidden_size];
      thread_in_grad = activation_bwd_kernel<act_type, float>(
          thread_in_grad * scale * static_cast<float>(dropout_mask[idx] & 1),
          in + b);
      thread_grad_bias += thread_in_grad;

      clip_bwd(thread_in_grad, temp_cmax_in_grad, thread_in_grad, cmask_in[idx],
               6);
      in_grad[idx] = thread_in_grad;
      thread_cmax_in_grad += temp_cmax_in_grad;
      idx += stride;
    }
  }
  __shared__ float block_cmax_in_grad, block_cmax_out_grad;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    block_cmax_in_grad = 0;
    block_cmax_out_grad = 0;
  }
  __syncthreads();

  if (thread_cmax_out_grad != 0) {
    atomicAdd(&block_cmax_out_grad, thread_cmax_out_grad);
  }
  if (thread_cmax_in_grad != 0) {
    atomicAdd(&block_cmax_in_grad, thread_cmax_in_grad);
  }

  tile[threadIdx.x][threadIdx.y] = thread_grad_bias;
  __syncthreads();
  float sum = tile[threadIdx.y][threadIdx.x];

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    if (block_cmax_in_grad != 0) {
      atomicAdd(&cmax_in_grad[0], block_cmax_in_grad);
    }
    if (block_cmax_out_grad != 0) {
      atomicAdd(&cmax_out_grad[0], block_cmax_out_grad);
    }
  }

  __syncthreads();

  for (int i = 1; i < WARP_SIZE; i <<= 1) sum += g.shfl_down(sum, i);

  if (threadIdx.x == 0) tile[0][threadIdx.y] = sum;
  __syncthreads();

  if (threadIdx.y == 0) {
    int pos = flat_2dim(blockIdx.x, threadIdx.x, WARP_SIZE);
    bias_grad[pos] = tile[0][threadIdx.x];
  }
}

template <ActivationType act_type, typename T>
void launch_ls_quant_dropout_act_bias_bwd(
    T *in_grad, T *bias_grad, T *cmax_in_grad, T *cmax_out_grad, const T *input,
    const T *cmax_in, const uint8_t *cmask_in, const uint8_t *cmask_out,
    const T *bias, const T *out_grad, const uint8_t *dropout_mask, int row_size,
    int dim, float ratio, cudaStream_t stream) {
  zero_grad<<<1, 1>>>(cmax_in_grad);
  zero_grad<<<1, 1>>>(cmax_out_grad);
  dim3 grid_dim((dim - 1) / WARP_SIZE + 1);
  dim3 block_dim(WARP_SIZE, WARP_SIZE);
  ls_quant_dropout_act_bias_bwd_kernel<act_type>
      <<<grid_dim, block_dim, 0, stream>>>(in_grad, bias_grad, cmax_in_grad,
                                           cmax_out_grad, input, cmax_in,
                                           cmask_in, cmask_out, bias, out_grad,
                                           dropout_mask, row_size, ratio, dim);
}

template void
launch_ls_quant_dropout_act_bias_bwd<ActivationType::kRelu, float>(
    float *in_grad, float *bias_grad, float *cmax_in_grad, float *cmax_out_grad,
    const float *input, const float *cmax_in, const uint8_t *cmask_in,
    const uint8_t *cmask_out, const float *bias, const float *out_grad,
    const uint8_t *dropout_mask, int row_size, int dim, float ratio,
    cudaStream_t stream);

template void
launch_ls_quant_dropout_act_bias_bwd<ActivationType::kRelu, __half>(
    __half *in_grad, __half *bias_grad, __half *cmax_in_grad,
    __half *cmax_out_grad, const __half *input, const __half *cmax_in,
    const uint8_t *cmask_in, const uint8_t *cmask_out, const __half *bias,
    const __half *out_grad, const uint8_t *dropout_mask, int row_size, int dim,
    float ratio, cudaStream_t stream);

template void
launch_ls_quant_dropout_act_bias_bwd<ActivationType::kGelu, float>(
    float *in_grad, float *bias_grad, float *cmax_in_grad, float *cmax_out_grad,
    const float *input, const float *cmax_in, const uint8_t *cmask_in,
    const uint8_t *cmask_out, const float *bias, const float *out_grad,
    const uint8_t *dropout_mask, int row_size, int dim, float ratio,
    cudaStream_t stream);

template void
launch_ls_quant_dropout_act_bias_bwd<ActivationType::kGelu, __half>(
    __half *in_grad, __half *bias_grad, __half *cmax_in_grad,
    __half *cmax_out_grad, const __half *input, const __half *cmax_in,
    const uint8_t *cmask_in, const uint8_t *cmask_out, const __half *bias,
    const __half *out_grad, const uint8_t *dropout_mask, int row_size, int dim,
    float ratio, cudaStream_t stream);

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
__global__ void ls_quant_dropout_res_bias_kernel(
    const int total_count, const float ratio, float *out, uint8_t *mask,
    const int8_t *qin, const float *cmax, const float *bias,
    const float *residual, const int seed, const int hidden_size,
    bool in_col32) {
  const float scale = 1.f / (1.f - ratio);
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 4 >= total_count) return;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, i, 0, &state);
  uint8_t m[4];

  float4 *out4 = reinterpret_cast<float4 *>(out);
  const int32_t *qdata4 = reinterpret_cast<const int32_t *>(qin);
  const float4 *residual4 = reinterpret_cast<const float4 *>(residual);
  const float4 *bias4 = reinterpret_cast<const float4 *>(bias);
  uint32_t *mask4 = reinterpret_cast<uint32_t *>(mask);
  float4 rand = curand_uniform4(&state);

  m[0] = static_cast<uint8_t>(rand.x > ratio);
  m[1] = static_cast<uint8_t>(rand.y > ratio);
  m[2] = static_cast<uint8_t>(rand.z > ratio);
  m[3] = static_cast<uint8_t>(rand.w > ratio);

  float cmax_val = cmax[0];
  int bias_i = i % (hidden_size >> 2);
  uint32_t *m4 = reinterpret_cast<uint32_t *>(m);
  mask4[i] |= m4[0];

  int input_index;
  if (in_col32) {
    int batch_tokens = total_count / hidden_size;
    int row_id = (i * 4) / hidden_size;
    int col_id = (i * 4) % hidden_size;
    input_index =
        row_major2flat_col32(row_id, col_id, batch_tokens, hidden_size) / 4;
  } else {
    input_index = i;
  }
  int32_t qinput4 = qdata4[input_index];
  int8_t *qinput = reinterpret_cast<int8_t *>(&qinput4);
  const float4 b4 = __ldg(&bias4[bias_i]);
  const float4 res4 = residual4[i];
  float4 output4;

  output4.x = (dequantize(qinput[0], cmax_val) + b4.x) * scale * m[0] + res4.x;
  output4.y = (dequantize(qinput[1], cmax_val) + b4.y) * scale * m[1] + res4.y;
  output4.z = (dequantize(qinput[2], cmax_val) + b4.z) * scale * m[2] + res4.z;
  output4.w = (dequantize(qinput[3], cmax_val) + b4.w) * scale * m[3] + res4.w;

  out4[i] = output4;
}

__global__ void ls_quant_dropout_res_bias_kernel(
    const int total_count, const float ratio, __half *out, uint8_t *mask,
    const int8_t *qin, const __half *cmax, const __half *bias,
    const __half *residual, const int seed, const int hidden_size,
    bool in_col32) {
  const __half scale = 1.f / (1.f - ratio);

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i * 8 >= total_count) return;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, i, 0, &state);

  const int64_t *qvals8_ptr = reinterpret_cast<const int64_t *>(qin);
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
  mask8[i] |= m8[0];

  float cmax_val = __half2float(cmax[0]);
  int bias_i = i % (hidden_size >> 3);
  int input_index;
  if (in_col32) {
    int batch_tokens = total_count / hidden_size;
    int row_id = (i * 8) / hidden_size;
    int col_id = (i * 8) % hidden_size;
    input_index =
        row_major2flat_col32(row_id, col_id, batch_tokens, hidden_size) / 8;
  } else {
    input_index = i;
  }
  int64_t qval8 = qvals8_ptr[input_index];
  int8_t *qval = reinterpret_cast<int8_t *>(&qval8);
  const float4 b4 = __ldg(&bias4[bias_i]);
  const float4 res4 = residual4[i];
  float4 out_float4;

  __half2 *out_half2 = reinterpret_cast<__half2 *>(&out_float4);
  const __half2 *b_half2 = reinterpret_cast<const __half2 *>(&b4);
  const __half2 *res_half2 = reinterpret_cast<const __half2 *>(&res4);

  __half2 scale_mask[4];
  scale_mask[0] =
      __halves2half2(scale * __float2half(m[0]), scale * __float2half(m[1]));
  scale_mask[1] =
      __halves2half2(scale * __float2half(m[2]), scale * __float2half(m[3]));
  scale_mask[2] =
      __halves2half2(scale * __float2half(m[4]), scale * __float2half(m[5]));
  scale_mask[3] =
      __halves2half2(scale * __float2half(m[6]), scale * __float2half(m[7]));

  float2 f_val;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    f_val.x = dequantize(qval[2 * j], cmax_val);
    f_val.y = dequantize(qval[2 * j + 1], cmax_val);
    out_half2[j] = __hfma2(__hadd2(__float22half2_rn(f_val), b_half2[j]),
                           scale_mask[j], res_half2[j]);
  }

  outs_float4[i] = out_float4;
}

template <>
void launch_ls_quant_dropout_res_bias<float>(
    float *out, uint8_t *mask, const int8_t *qvals, const float *cmax,
    const float *bias, const float *residual, int total_count, int dim,
    float ratio, cudaStream_t stream, bool in_col32) {
  int grid_dim = total_count >> 12;
  ls_quant_dropout_res_bias_kernel<<<grid_dim + 1, 1024, 0, stream>>>(
      total_count, ratio, out, mask, qvals, cmax, bias, residual,
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count(),
      dim, in_col32);
}

template <>
void launch_ls_quant_dropout_res_bias<__half>(
    __half *out, uint8_t *mask, const int8_t *qvals, const __half *cmax,
    const __half *bias, const __half *residual, int total_count, int dim,
    float ratio, cudaStream_t stream, bool in_col32) {
  int grid_dim = total_count >> 13;
  ls_quant_dropout_res_bias_kernel<<<grid_dim + 1, 1024, 0, stream>>>(
      total_count, ratio, out, mask, qvals, cmax, bias, residual,
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count(),
      dim, in_col32);
}
}  // namespace cuda
}  // namespace lightseq
