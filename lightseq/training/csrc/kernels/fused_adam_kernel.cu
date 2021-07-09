/* Copyright 2021 The LightSeq Team
   Copyright NVIDIA/apex
   This apex_adam_cuda_kernel is adapted from NVIDIA/apex
*/
#include <THC/THCGeneral.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cmath>

#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/TensorUtils.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/detail/IndexUtils.cuh"
#include "fused_adam_kernel.h"
#include "multi_tensor_apply.cuh"

typedef enum {
  ADAM_MODE_0 = 0,  // eps under square root
  ADAM_MODE_1 = 1   // eps outside square root
} adamMode_t;

template <typename T, typename GRAD_T>
__global__ void apex_adam_cuda_kernel(
    T* __restrict__ p,
    GRAD_T* __restrict__ p_copy,  // For mixed precision training, pass NULL if
                                  // not needed
    T* __restrict__ m, T* __restrict__ v, const GRAD_T* __restrict__ g,
    const float b1, const float b2, const float eps, const float grad_scale,
    const float step_size, const size_t tsize, adamMode_t mode,
    const float decay) {
  // Assuming 2D grids and 2D blocks
  const int blockId = gridDim.x * blockIdx.y + blockIdx.x;
  const int block_dim = blockDim.x * blockDim.y;
  const int threadIdInBlock = threadIdx.y * blockDim.x + threadIdx.x;
  const int i = (blockId * block_dim + threadIdInBlock);
  const int totThreads = gridDim.x * gridDim.y * block_dim;
  for (int j = i; j < tsize; j += totThreads) {
    T scaled_grad = g[j] / grad_scale;
    m[j] = b1 * m[j] + (1 - b1) * scaled_grad;
    v[j] = b2 * v[j] + (1 - b2) * scaled_grad * scaled_grad;
    float denom;
    if (mode == ADAM_MODE_0)
      denom = sqrtf(v[j] + eps);
    else  // Mode 1
      denom = sqrtf(v[j]) + eps;
    float update = (m[j] / denom) + (decay * p[j]);
    p[j] = p[j] - (step_size * update);
    if (p_copy != NULL) {
      p_copy[j] = (GRAD_T)p[j];
    }
  }
}

template <typename T, typename GRAD_T>
__global__ void ls_adam_cuda_kernel(
    T* __restrict__ p,
    GRAD_T* __restrict__ p_copy,  // For mixed precision training, pass NULL if
                                  // not needed
    T* __restrict__ m, T* __restrict__ v, const GRAD_T* __restrict__ g,
    const float b1, const float b2, const float eps, const float grad_scale,
    const float step_size, const size_t total_size, adamMode_t mode,
    const float decay) {
  int global_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_id >= total_size) return;

  T scaled_grad = g[global_id] / grad_scale;
  m[global_id] = b1 * m[global_id] + (1 - b1) * scaled_grad;
  v[global_id] = b2 * v[global_id] + (1 - b2) * scaled_grad * scaled_grad;
  float denom;
  if (mode == ADAM_MODE_0)
    denom = sqrtf(v[global_id] + eps);
  else  // Mode 1
    denom = sqrtf(v[global_id]) + eps;
  float update = (m[global_id] / denom) + (decay * p[global_id]);
  p[global_id] = p[global_id] - (step_size * update);
  if (p_copy != NULL) p_copy[global_id] = (GRAD_T)p[global_id];
}

template <>
__global__ void ls_adam_cuda_kernel<float, float>(
    float* __restrict__ p,
    float* __restrict__ p_copy,  // For mixed precision training, pass NULL if
                                 // not needed
    float* __restrict__ m, float* __restrict__ v, const float* __restrict__ g,
    const float b1, const float b2, const float eps, const float grad_scale,
    const float step_size, const size_t total_size, adamMode_t mode,
    const float decay) {
  int global_id = (blockIdx.x * blockDim.x + threadIdx.x);

  if (global_id * 4 >= total_size) return;

  const float4* g4_ptr = reinterpret_cast<const float4*>(g);
  float4* p4_ptr = reinterpret_cast<float4*>(p);
  float4* m4_ptr = reinterpret_cast<float4*>(m);
  float4* v4_ptr = reinterpret_cast<float4*>(v);

  const float4 g4 = g4_ptr[global_id];
  const float4 p4 = p4_ptr[global_id];
  const float4 m4 = m4_ptr[global_id];
  const float4 v4 = v4_ptr[global_id];

  float4 new_p4;
  float4 new_m4;
  float4 new_v4;

  float scaled_grad1 = g4.x / grad_scale;
  float scaled_grad2 = g4.y / grad_scale;
  float scaled_grad3 = g4.z / grad_scale;
  float scaled_grad4 = g4.w / grad_scale;

  new_m4.x = b1 * m4.x + (1 - b1) * scaled_grad1;
  new_m4.y = b1 * m4.y + (1 - b1) * scaled_grad2;
  new_m4.z = b1 * m4.z + (1 - b1) * scaled_grad3;
  new_m4.w = b1 * m4.w + (1 - b1) * scaled_grad4;

  new_v4.x = b2 * v4.x + (1 - b2) * scaled_grad1 * scaled_grad1;
  new_v4.y = b2 * v4.y + (1 - b2) * scaled_grad2 * scaled_grad2;
  new_v4.z = b2 * v4.z + (1 - b2) * scaled_grad3 * scaled_grad3;
  new_v4.w = b2 * v4.w + (1 - b2) * scaled_grad4 * scaled_grad4;

  float4 denom4;

  denom4.x =
      mode == ADAM_MODE_0 ? sqrtf(new_v4.x + eps) : sqrtf(new_v4.x) + eps;
  denom4.y =
      mode == ADAM_MODE_0 ? sqrtf(new_v4.y + eps) : sqrtf(new_v4.y) + eps;
  denom4.z =
      mode == ADAM_MODE_0 ? sqrtf(new_v4.z + eps) : sqrtf(new_v4.z) + eps;
  denom4.w =
      mode == ADAM_MODE_0 ? sqrtf(new_v4.w + eps) : sqrtf(new_v4.w) + eps;

  new_p4.x = p4.x - (step_size * (new_m4.x / denom4.x + decay * p4.x));
  new_p4.y = p4.y - (step_size * (new_m4.y / denom4.y + decay * p4.y));
  new_p4.z = p4.z - (step_size * (new_m4.z / denom4.z + decay * p4.z));
  new_p4.w = p4.w - (step_size * (new_m4.w / denom4.w + decay * p4.w));

  p4_ptr[global_id] = new_p4;
  m4_ptr[global_id] = new_m4;
  v4_ptr[global_id] = new_v4;
}

void fused_adam_cuda(at::Tensor& p, at::Tensor& p_copy, at::Tensor& m,
                     at::Tensor& v, at::Tensor& g, float lr, float beta1,
                     float beta2, float eps, float grad_scale, int step,
                     int mode, int bias_correction, float decay) {
  // Get tensor size
  int total_size = p.numel();
  AT_ASSERTM(at::cuda::detail::canUse32BitIndexMath(p),
             "parameter tensor is too large to be indexed with int32");
  // Constants
  float step_size = 0;
  if (bias_correction == 1) {
    const float bias_correction1 = 1 - std::pow(beta1, step);
    const float bias_correction2 = 1 - std::pow(beta2, step);
    step_size = lr * std::sqrt(bias_correction2) / bias_correction1;
  } else {
    step_size = lr;
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (g.scalar_type() == at::ScalarType::Half) {
    const int block_dim = 1024;
    int grid_dim = ((total_size + block_dim - 1) / block_dim);
    const dim3 blocks(grid_dim);
    // all other values should be fp32 for half gradients
    AT_ASSERTM(p.scalar_type() == at::ScalarType::Float,
               "expected parameter to be of float type");
    // dispatch is done on the gradient type
    using namespace at;  // prevents "toString is undefined" errors
    DISPATCH_FLOAT_AND_HALF(
        g.scalar_type(), 0, "adam_cuda_kernel",
        using accscalar_t = at::acc_type<scalar_t_0, true>;
        ls_adam_cuda_kernel<accscalar_t, scalar_t_0>
        <<<blocks, block_dim, 0, stream>>>(
            p.DATA_PTR<accscalar_t>(),
            p_copy.numel() ? p_copy.DATA_PTR<scalar_t_0>() : NULL,
            m.DATA_PTR<accscalar_t>(), v.DATA_PTR<accscalar_t>(),
            g.DATA_PTR<scalar_t_0>(), beta1, beta2, eps, grad_scale, step_size,
            total_size, (adamMode_t)mode, decay););
  } else {
    using namespace at;
    const int block_dim = 1024;
    int grid_dim = ((total_size + block_dim - 1) / block_dim) >> 2;
    if (grid_dim == 0) grid_dim = 1;
    const dim3 blocks(grid_dim);
    DISPATCH_DOUBLE_AND_FLOAT(
        g.scalar_type(), 0, "adam_cuda_kernel",
        ls_adam_cuda_kernel<scalar_t_0, scalar_t_0>
        <<<blocks, block_dim, 0, stream>>>(
            p.DATA_PTR<scalar_t_0>(),
            NULL,  // don't output p_copy for fp32, it's wasted write
            m.DATA_PTR<scalar_t_0>(), v.DATA_PTR<scalar_t_0>(),
            g.DATA_PTR<scalar_t_0>(), beta1, beta2, eps, grad_scale, step_size,
            total_size, (adamMode_t)mode, decay););
  }
  THCudaCheck(cudaGetLastError());
}

void apex_fused_adam_cuda(at::Tensor& p, at::Tensor& p_copy, at::Tensor& m,
                          at::Tensor& v, at::Tensor& g, float lr, float beta1,
                          float beta2, float eps, float grad_scale, int step,
                          int mode, int bias_correction, float decay) {
  // Get tensor size
  int tsize = p.numel();
  // Determine #threads and #blocks
  const int threadsPerBlock = 512;
  const dim3 blocks((tsize + threadsPerBlock - 1) / threadsPerBlock);
  AT_ASSERTM(at::cuda::detail::canUse32BitIndexMath(p),
             "parameter tensor is too large to be indexed with int32");
  // Constants
  float step_size = 0;
  if (bias_correction == 1) {
    const float bias_correction1 = 1 - std::pow(beta1, step);
    const float bias_correction2 = 1 - std::pow(beta2, step);
    step_size = lr * std::sqrt(bias_correction2) / bias_correction1;
  } else {
    step_size = lr;
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (g.scalar_type() == at::ScalarType::Half) {
    // all other values should be fp32 for half gradients
    AT_ASSERTM(p.scalar_type() == at::ScalarType::Float,
               "expected parameter to be of float type");
    // dispatch is done on the gradient type
    using namespace at;  // prevents "toString is undefined" errors
    DISPATCH_FLOAT_AND_HALF(
        g.scalar_type(), 0, "apex_adam_cuda_kernel",
        using accscalar_t = at::acc_type<scalar_t_0, true>;
        apex_adam_cuda_kernel<accscalar_t, scalar_t_0>
        <<<blocks, threadsPerBlock, 0, stream>>>(
            p.DATA_PTR<accscalar_t>(),
            p_copy.numel() ? p_copy.DATA_PTR<scalar_t_0>() : NULL,
            m.DATA_PTR<accscalar_t>(), v.DATA_PTR<accscalar_t>(),
            g.DATA_PTR<scalar_t_0>(), beta1, beta2, eps, grad_scale, step_size,
            tsize, (adamMode_t)mode, decay););
  } else {
    using namespace at;
    DISPATCH_DOUBLE_AND_FLOAT(
        g.scalar_type(), 0, "apex_adam_cuda_kernel",
        apex_adam_cuda_kernel<scalar_t_0, scalar_t_0>
        <<<blocks, threadsPerBlock, 0, stream>>>(
            p.DATA_PTR<scalar_t_0>(),
            NULL,  // don't output p_copy for fp32, it's wasted write
            m.DATA_PTR<scalar_t_0>(), v.DATA_PTR<scalar_t_0>(),
            g.DATA_PTR<scalar_t_0>(), beta1, beta2, eps, grad_scale, step_size,
            tsize, (adamMode_t)mode, decay););
  }
  THCudaCheck(cudaGetLastError());
}
