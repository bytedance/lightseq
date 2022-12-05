#ifdef __HIPCC__
#include <hip_cooperative_groups.h>
#include "./hip/block_reduce_hip.h"
#else
#include <cooperative_groups.h>
#include "block_reduce.h"
#endif
#include "kernels.h"
namespace cg = cooperative_groups;
const float LN_EPSILON = 1e-8f;
#define TILE_DIM 32

template <typename T>
__forceinline__ __device__ T add_eps(T x) {
  return fabsf(x) > LN_EPSILON ? x : (x < 0 ? -LN_EPSILON : LN_EPSILON);
}

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
__global__ void ker_layer_norm(T *ln_res, T *vars, T *means, const T *inp,
                               const T *scale, const T *bias, int hidden_size) {
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
  float4 *output_f4 = (float4 *)ln_res + blockIdx.x * hidden_size;
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 vscale = __ldg((const float4 *)scale + idx);
    float4 vbias = __ldg((const float4 *)bias + idx);
    float4 val = inp_f4[idx];
    val.x = (val.x - s_mean) * s_var * vscale.x + vbias.x;
    val.y = (val.y - s_mean) * s_var * vscale.y + vbias.y;
    val.z = (val.z - s_mean) * s_var * vscale.z + vbias.z;
    val.w = (val.w - s_mean) * s_var * vscale.w + vbias.w;
    output_f4[idx] = val;
  }
}

template <>
__global__ void ker_layer_norm<__half>(__half *ln_res, __half *vars,
                                       __half *means, const __half *inp,
                                       const __half *scale, const __half *bias,
                                       int hidden_size) {
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
  float4 *output_f4 = (float4 *)ln_res + blockIdx.x * hidden_size;
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    // load scale, bias, input
    float4 scale_f4 = __ldg((const float4 *)scale + idx);
    __half2 *scale_h2 = (__half2 *)(&scale_f4);
    float4 bias_f4 = __ldg((const float4 *)bias + idx);
    __half2 *bias_h2 = (__half2 *)(&bias_f4);
    float4 val_f4 = inp_f4[idx];
    __half2 *val_h2 = (__half2 *)(&val_f4);

#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 scale_f2 = __half22float2(scale_h2[i]);
      float2 bias_f2 = __half22float2(bias_h2[i]);
      float2 val_f2 = __half22float2(val_h2[i]);
      val_f2.x = (val_f2.x - s_mean) * s_var * scale_f2.x + bias_f2.x;
      val_f2.y = (val_f2.y - s_mean) * s_var * scale_f2.y + bias_f2.y;
      val_h2[i] = __float22half2_rn(val_f2);
    }
    output_f4[idx] = val_f4;
  }
}

template <>
void launch_layer_norm<float>(float *ln_res, float *vars, float *means,
                              const float *inp, const float *scale,
                              const float *bias, int batch_size, int hidden_dim,
                              cudaStream_t stream) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  hidden_dim >>= 2;
// aiss add
#ifdef __HIPCC__
  int nthread = 1024;
#else
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
#endif
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm<float><<<grid_dim, block_dim, 0, stream>>>(
      ln_res, vars, means, inp, scale, bias, hidden_dim);
}

template <>
void launch_layer_norm<__half>(__half *ln_res, __half *vars, __half *means,
                               const __half *inp, const __half *scale,
                               const __half *bias, int batch_size,
                               int hidden_dim, cudaStream_t stream) {
  if (hidden_dim % 8 != 0) {
    throw std::runtime_error("violate hidden_dim % 8 = 0");
  }
  hidden_dim >>= 3;
#ifdef __HIPCC__
  int nthread = 1024;
#else
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
#endif
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm<__half><<<grid_dim, block_dim, 0, stream>>>(
      ln_res, vars, means, inp, scale, bias, hidden_dim);
}

/**
@brief: ker_ln_bw_dgamma_dbetta
Layer norm backword kernel, compute the gradient of gamma and betta.
dbetta = sum(dout, dim=0)
dgamma = sum(xhat * dout, dim=0)
xhat = (input - mean) * rsqrt(var) or
  (output - betta) / gamma


@thread
gridDim.x = hidden_size / 32
blockDim.x = 32
blockDim.y = 32

@param
gamma_grad: [hidden_size], gradient of gamma
betta_grad: [hidden_size], gradient of betta
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat, maybe nullptr
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat, maybe nullptr
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
(gamma && betta) ^ (vars && means) should be true
*/
template <typename T>
__global__ void ker_ln_bw_dgamma_dbetta(T *gamma_grad, T *betta_grad,
                                        const T *out_grad, const T *inp_or_out,
                                        const T *gamma, const T *betta,
                                        const T *vars, const T *means, int rows,
                                        int width) {
  __shared__ float betta_buffer[TILE_DIM][TILE_DIM];
  __shared__ float gamma_buffer[TILE_DIM][TILE_DIM];
#ifndef __HIPCC__
  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);
#endif
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int offset = threadIdx.y * width + idx;
  int y_stride = width * TILE_DIM;

  // Loop across inp height
  float dbetta = 0;
  float dgamma = 0;
  float dout, val;
  if (idx < width) {
    if (means == nullptr) {
      float vbetta = (float)betta[idx];
      float vgamma = (float)gamma[idx];
      for (int r = threadIdx.y; r < rows; r += TILE_DIM) {
        dout = (float)out_grad[offset];
        // inp_or_out is output
        val = (float)inp_or_out[offset];
        dbetta += dout;
        dgamma += ((val - vbetta) / add_eps(vgamma) * dout);
        offset += y_stride;
      }
    } else {
      for (int r = threadIdx.y; r < rows; r += TILE_DIM) {
        dout = (float)out_grad[offset];
        // inp_or_out is input
        val = (float)inp_or_out[offset];
        dbetta += dout;
        dgamma += ((val - (float)means[r]) *
                   rsqrtf((float)vars[r] + LN_EPSILON) * dout);
        offset += y_stride;
      }
    }
  }

  // Sum the shared buffer.
  betta_buffer[threadIdx.x][threadIdx.y] = dbetta;
  gamma_buffer[threadIdx.x][threadIdx.y] = dgamma;
  __syncthreads();
  float s1 = betta_buffer[threadIdx.y][threadIdx.x];
  float s2 = gamma_buffer[threadIdx.y][threadIdx.x];
  __syncthreads();

  for (int i = 1; i < TILE_DIM; i <<= 1) {
#ifdef __HIPCC__
    s1 += __shfl_down(s1, i, 32);
    s2 += __shfl_down(s2, i, 32);
#else
    s1 += g.shfl_down(s1, i);
    s2 += g.shfl_down(s2, i);
#endif
  }

  int pos = blockIdx.x * TILE_DIM + threadIdx.y;
  if (threadIdx.x == 0 && idx < width) {
    betta_grad[pos] = s1;
    gamma_grad[pos] = s2;
  }
}

#ifdef __HIPCC__
template <>
__global__ void ker_ln_bw_dgamma_dbetta<__half>(
    __half *gamma_grad, __half *betta_grad, const __half *out_grad,
    const __half *inp_or_out, const __half *gamma, const __half *betta,
    const __half *vars, const __half *means, int rows, int width)
    __attribute__((amdgpu_flat_work_group_size(1, 1024))) {
  __shared__ float betta_buffer[TILE_DIM][TILE_DIM];
  __shared__ float gamma_buffer[TILE_DIM][TILE_DIM];

  // cg::thread_block b = cg::this_thread_block();
  // cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int offset = threadIdx.y * width + idx;
  int y_stride = width * TILE_DIM;

  // Loop across inp height
  float dbetta = 0;
  float dgamma = 0;
  float dout, val;
  if (idx < width) {
    if (means == nullptr) {
      float vbetta = __half2float(betta[idx]);
      float vgamma = __half2float(gamma[idx]);
      for (int r = threadIdx.y; r < rows; r += TILE_DIM) {
        dout = __half2float(out_grad[offset]);
        // inp_or_out is output
        val = __half2float(inp_or_out[offset]);
        dbetta += dout;
        dgamma += ((val - vbetta) / add_eps(vgamma) * dout);
        offset += y_stride;
      }
    } else {
      for (int r = threadIdx.y; r < rows; r += TILE_DIM) {
        dout = __half2float(out_grad[offset]);
        // inp_or_out is input
        val = __half2float(inp_or_out[offset]);
        dbetta += dout;
        dgamma += ((val - __half2float(means[r])) *
                   rsqrtf(__half2float(vars[r]) + LN_EPSILON) * dout);
        offset += y_stride;
      }
    }
  }

  // Sum the shared buffer.
  betta_buffer[threadIdx.x][threadIdx.y] = dbetta;
  gamma_buffer[threadIdx.x][threadIdx.y] = dgamma;
  __syncthreads();
  float s1 = betta_buffer[threadIdx.y][threadIdx.x];
  float s2 = gamma_buffer[threadIdx.y][threadIdx.x];
  __syncthreads();

  for (int i = 1; i < TILE_DIM; i <<= 1) {
    s1 += __shfl_down(s1, i, 32);
    s2 += __shfl_down(s2, i, 32);
  }
  __syncthreads();
  int pos = blockIdx.x * TILE_DIM + threadIdx.y;
  if (threadIdx.x == 0 && idx < width) {
    betta_grad[pos] = __float2half(s1);
    gamma_grad[pos] = __float2half(s2);
  }
}
#endif
/**
@brief: ker_ln_bw_dinp
Layer norm backword kernel, compute the gradient of input.
dinp = (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / hidden_dim)
  * rsqrt(var)
xhat = (input - mean) * rsqrt(var) if mean is not nullptr
       (output - betta) / gamma if mean is nullptr
dxhat = dout * gamma


@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
inp_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
residual_grad: [batch_size * seq_len, hidden_size], gradient of residual input,
  usually appear in pre-layer-norm for transformer layer, maybe nullptr
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat and dxhat
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat and dinp
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
*/
template <typename T>
__global__ void ker_ln_bw_dinp(T *inp_grad, const T *out_grad,
                               const T *residual_grad, const T *inp_or_out,
                               const T *gamma, const T *betta, const T *vars,
                               const T *means, int hidden_dim) {
  int offset = blockIdx.x * hidden_dim + threadIdx.x;
  float4 dxhat, xhat;
  float var_rsqrt;

  if (threadIdx.x < hidden_dim) {
    // step 0. dxhat = dout * gamma
    dxhat = ((const float4 *)out_grad)[offset];
    float4 vgamma = ((const float4 *)gamma)[threadIdx.x];
    dxhat.x *= vgamma.x;
    dxhat.y *= vgamma.y;
    dxhat.z *= vgamma.z;
    dxhat.w *= vgamma.w;

    /*
    step 1. xhat = (output - betta) / gamma or
    (input - mean) * rsqrtf(var)
    */
    xhat = ((const float4 *)inp_or_out)[offset];
    var_rsqrt = rsqrtf((float)vars[blockIdx.x] + LN_EPSILON);
    if (means == nullptr) {
      // inp_or_out is output, xhat = (output - betta) / gamma
      float4 vbetta = ((const float4 *)betta)[threadIdx.x];
      xhat.x = (xhat.x - vbetta.x) / add_eps(vgamma.x);
      xhat.y = (xhat.y - vbetta.y) / add_eps(vgamma.y);
      xhat.z = (xhat.z - vbetta.z) / add_eps(vgamma.z);
      xhat.w = (xhat.w - vbetta.w) / add_eps(vgamma.w);
    } else {
      // inp_or_out is input, xhat = (input - mean) * rsqrtf(var)
      float fmean = (float)means[blockIdx.x];
      xhat.x = (xhat.x - fmean) * var_rsqrt;
      xhat.y = (xhat.y - fmean) * var_rsqrt;
      xhat.z = (xhat.z - fmean) * var_rsqrt;
      xhat.w = (xhat.w - fmean) * var_rsqrt;
    }
  }

  /* step2. block reduce sum for dxhat and dxhat*xhat */
  float reduce_val[2] = {0.f, 0.f};
  if (threadIdx.x < hidden_dim) {
    reduce_val[0] = dxhat.x + dxhat.y + dxhat.z + dxhat.w;
    reduce_val[1] = dxhat.x * xhat.x + dxhat.y * xhat.y + dxhat.z * xhat.z +
                    dxhat.w * xhat.w;
  }
  blockReduce<ReduceType::kSum, 2>(reduce_val);
  __shared__ float s_sum_dxhat, s_sum_dxhat_xhat;
  if (threadIdx.x == 0) {
    float mean_dim = hidden_dim * 4;
    s_sum_dxhat = reduce_val[0] / mean_dim;
    s_sum_dxhat_xhat = reduce_val[1] / mean_dim;
  }
  __syncthreads();

  /*
  step3. compute input gradient
  (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / mean_dim) * rsqrt(var)
  */
  if (threadIdx.x >= hidden_dim) {
    return;
  }
  dxhat.x = (dxhat.x - s_sum_dxhat - xhat.x * s_sum_dxhat_xhat) * var_rsqrt;
  dxhat.y = (dxhat.y - s_sum_dxhat - xhat.y * s_sum_dxhat_xhat) * var_rsqrt;
  dxhat.z = (dxhat.z - s_sum_dxhat - xhat.z * s_sum_dxhat_xhat) * var_rsqrt;
  dxhat.w = (dxhat.w - s_sum_dxhat - xhat.w * s_sum_dxhat_xhat) * var_rsqrt;
  if (residual_grad) {
    // Add the residual grad,
    // usually in pre-layer-norm for transformer layer
    float4 dresidual = ((const float4 *)residual_grad)[offset];
    dxhat.x += dresidual.x;
    dxhat.y += dresidual.y;
    dxhat.z += dresidual.z;
    dxhat.w += dresidual.w;
  }
  ((float4 *)inp_grad)[offset] = dxhat;
}

template <>
__global__ void ker_ln_bw_dinp<__half>(__half *inp_grad, const __half *out_grad,
                                       const __half *residual_grad,
                                       const __half *inp_or_out,
                                       const __half *gamma, const __half *betta,
                                       const __half *vars, const __half *means,
                                       int hidden_dim) {
  int offset = blockIdx.x * hidden_dim + threadIdx.x;

  float2 dxhat[4], xhat[4];
  float var_rsqrt;
  float4 vtmp;
  __half2 *tmp_h2;
  float reduce_val[2] = {0.f, 0.f};

  if (threadIdx.x < hidden_dim) {
    // step 0. dxhat = dout * gamma
    vtmp = ((const float4 *)out_grad)[offset];
    tmp_h2 = reinterpret_cast<__half2 *>(&vtmp);
    float4 gamma_f4 = ((const float4 *)gamma)[threadIdx.x];
    __half2 *gamma_h2 = reinterpret_cast<__half2 *>(&gamma_f4);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 vdout = __half22float2(tmp_h2[i]);
      float2 vgamma = __half22float2(gamma_h2[i]);
      dxhat[i].x = vdout.x * vgamma.x;
      dxhat[i].y = vdout.y * vgamma.y;
      reduce_val[0] += dxhat[i].x + dxhat[i].y;
    }

    /*
    step 1. xhat = (output - betta) / gamma or
    (input - mean) * rsqrtf(var)
    */
    vtmp = ((const float4 *)inp_or_out)[offset];
    var_rsqrt = rsqrtf((float)vars[blockIdx.x] + LN_EPSILON);
    if (means == nullptr) {
      // inp_or_out is output, xhat = (output - betta) / gamma
      float4 vbetta = ((const float4 *)betta)[threadIdx.x];
      __half2 *betta_h2 = reinterpret_cast<__half2 *>(&vbetta);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float2 vout = __half22float2(tmp_h2[i]);
        float2 vgamma = __half22float2(gamma_h2[i]);
        float2 vbetta = __half22float2(betta_h2[i]);
        xhat[i].x = (vout.x - vbetta.x) / add_eps(vgamma.x);
        xhat[i].y = (vout.y - vbetta.y) / add_eps(vgamma.y);
        reduce_val[1] += xhat[i].x * dxhat[i].x + xhat[i].y * dxhat[i].y;
      }
    } else {
      // inp_or_out is input, xhat = (input - mean) * rsqrtf(var)
      float fmean = (float)means[blockIdx.x];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float2 vinp = __half22float2(tmp_h2[i]);
        xhat[i].x = (vinp.x - fmean) * var_rsqrt;
        xhat[i].y = (vinp.y - fmean) * var_rsqrt;
        reduce_val[1] += xhat[i].x * dxhat[i].x + xhat[i].y * dxhat[i].y;
      }
    }
  }

  /* step2. block reduce sum for dxhat and dxhat*xhat */
  blockReduce<ReduceType::kSum, 2>(reduce_val);
  __shared__ float s_sum_dxhat, s_sum_dxhat_xhat;
  if (threadIdx.x == 0) {
    float mean_dim = hidden_dim * 8;
    s_sum_dxhat = reduce_val[0] / mean_dim;
    s_sum_dxhat_xhat = reduce_val[1] / mean_dim;
  }
  __syncthreads();

  /*
  step3. compute input gradient
  (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / mean_dim) * rsqrt(var)
  */
  if (threadIdx.x >= hidden_dim) {
    return;
  }
  if (residual_grad) {
    // Add the residual grad,
    // usually in pre-layer-norm for transformer layer
    float4 dresidual = ((const float4 *)residual_grad)[offset];
    __half *hdres = reinterpret_cast<__half *>(&dresidual);
#pragma unroll
    for (int i = 0; i < 4; i++) {
#ifdef __HIPCC__
      float tmpf_x = (dxhat[i].x - s_sum_dxhat - xhat[i].x * s_sum_dxhat_xhat) *
                         var_rsqrt +
                     __half2float(hdres[2 * i]);
      float tmpf_y = (dxhat[i].y - s_sum_dxhat - xhat[i].y * s_sum_dxhat_xhat) *
                         var_rsqrt +
                     __half2float(hdres[2 * i + 1]);
      float2 tmpf = float2(tmpf_x, tmpf_y);
      tmp_h2[i] = __float22half2_rn(tmpf);
#else
      tmp_h2[i].x = __float2half(
          (dxhat[i].x - s_sum_dxhat - xhat[i].x * s_sum_dxhat_xhat) *
              var_rsqrt +
          __half2float(hdres[2 * i]));
      tmp_h2[i].y = __float2half(
          (dxhat[i].y - s_sum_dxhat - xhat[i].y * s_sum_dxhat_xhat) *
              var_rsqrt +
          __half2float(hdres[2 * i + 1]));
#endif
    }
  } else {
#pragma unroll
    for (int i = 0; i < 4; i++) {
#ifdef __HIPCC__
      float tmpf_x =
          (dxhat[i].x - s_sum_dxhat - xhat[i].x * s_sum_dxhat_xhat) * var_rsqrt;
      float tmpf_y =
          (dxhat[i].y - s_sum_dxhat - xhat[i].y * s_sum_dxhat_xhat) * var_rsqrt;
      float2 tmpf = float2(tmpf_x, tmpf_y);
      tmp_h2[i] = __float22half2_rn(tmpf);
#else
      tmp_h2[i].x = __float2half(
          (dxhat[i].x - s_sum_dxhat - xhat[i].x * s_sum_dxhat_xhat) *
          var_rsqrt);
      tmp_h2[i].y = __float2half(
          (dxhat[i].y - s_sum_dxhat - xhat[i].y * s_sum_dxhat_xhat) *
          var_rsqrt);
#endif
    }
  }
  ((float4 *)inp_grad)[offset] = vtmp;
}

/**
Layer norm backword,
  compute the gradient of gamma, betta and input.
dbetta = sum(dout, dim=0)
xhat = (input - mean) * rsqrt(var) if mean is not nullptr
  (output - betta) / gamma if mean is nullptr
dgamma = sum(xhat * dout, dim=0)
dxhat = dout * gamma
dinp = (dxhat - (sum(dxhat, 1) + xhat * sum(dxhat * xhat, 1)) / hidden_dim)
  * rsqrt(var)

residual_grad, means, betta can be nullptr.
residual_grad will be added to dinp if it is not nullptr
  which is useful in transformer layer when pre-ln
means and betta are only used to compute xhat,
  (means == nullptr) ^ (betta == nullptr) should be true
*/
template <>
void launch_ln_bw<float>(float *gamma_grad, float *betta_grad, float *inp_grad,
                         const float *out_grad, const float *residual_grad,
                         const float *inp_or_out, const float *gamma,
                         const float *betta, const float *vars,
                         const float *means, int batch, int hidden_dim,
                         cudaStream_t stream[2]) {
  // compute grad of gamma and betta
  dim3 grid_dim(((hidden_dim + TILE_DIM - 1) / TILE_DIM) * TILE_DIM);
  dim3 block_dim(TILE_DIM, TILE_DIM);
  ker_ln_bw_dgamma_dbetta<float><<<grid_dim, block_dim, 0, stream[0]>>>(
      gamma_grad, betta_grad, out_grad, inp_or_out, gamma, betta, vars, means,
      batch, hidden_dim);

  // compute grad of input
  if (hidden_dim % 4 != 0 || hidden_dim > 4096) {
    throw std::runtime_error("hidden_dim % 4 != 0 || hidden_dim > 4096");
  }
  hidden_dim >>= 2;
#ifdef __HIPCC__
  int nthread = 1024;
#else
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
#endif
  ker_ln_bw_dinp<<<batch, nthread, 0, stream[1]>>>(
      inp_grad, out_grad, residual_grad, inp_or_out, gamma, betta, vars, means,
      hidden_dim);
}

template <>
void launch_ln_bw<__half>(__half *gamma_grad, __half *betta_grad,
                          __half *inp_grad, const __half *out_grad,
                          const __half *residual_grad, const __half *inp_or_out,
                          const __half *gamma, const __half *betta,
                          const __half *vars, const __half *means, int batch,
                          int hidden_dim, cudaStream_t stream[2]) {
  // compute grad of gamma and betta
  dim3 grid_dim(((hidden_dim + TILE_DIM - 1) / TILE_DIM) * TILE_DIM);
  dim3 block_dim(TILE_DIM, TILE_DIM);
  ker_ln_bw_dgamma_dbetta<__half><<<grid_dim, block_dim, 0, stream[0]>>>(
      gamma_grad, betta_grad, out_grad, inp_or_out, gamma, betta, vars, means,
      batch, hidden_dim);

  // compute grad of input
  if (hidden_dim % 8 != 0 || hidden_dim > 8192) {
    throw std::runtime_error("hidden_dim % 8 != 0 || hidden_dim > 8192");
  }
  hidden_dim >>= 3;
#ifdef __HIPCC__
  int nthread = 1024;
#else
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
#endif
  ker_ln_bw_dinp<<<batch, nthread, 0, stream[1]>>>(
      inp_grad, out_grad, residual_grad, inp_or_out, gamma, betta, vars, means,
      hidden_dim);
}
