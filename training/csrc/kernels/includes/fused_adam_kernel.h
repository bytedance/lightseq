/* Copyright 2021 The LightSeq Team
   Copyright NVIDIA/apex
   This apex_adam_cuda_kernel is adapted from NVIDIA/apex
*/
#include <ATen/ATen.h>
#include <torch/extension.h>

#define DISPATCH_FLOAT_AND_HALF(TYPE, LEVEL, NAME, ...)               \
  switch (TYPE) {                                                     \
    case at::ScalarType::Float: {                                     \
      using scalar_t_##LEVEL = float;                                 \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case at::ScalarType::Half: {                                      \
      using scalar_t_##LEVEL = at::Half;                              \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    default:                                                          \
      AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
  }

#define DISPATCH_DOUBLE_FLOAT_AND_HALF(TYPE, LEVEL, NAME, ...)        \
  switch (TYPE) {                                                     \
    case at::ScalarType::Double: {                                    \
      using scalar_t_##LEVEL = double;                                \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case at::ScalarType::Float: {                                     \
      using scalar_t_##LEVEL = float;                                 \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case at::ScalarType::Half: {                                      \
      using scalar_t_##LEVEL = at::Half;                              \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    default:                                                          \
      AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
  }

#define DISPATCH_DOUBLE_AND_FLOAT(TYPE, LEVEL, NAME, ...)             \
  switch (TYPE) {                                                     \
    case at::ScalarType::Double: {                                    \
      using scalar_t_##LEVEL = double;                                \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case at::ScalarType::Float: {                                     \
      using scalar_t_##LEVEL = float;                                 \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    default:                                                          \
      AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
  }

// CUDA forward declaration
void fused_adam_cuda(at::Tensor& p, at::Tensor& p_copy, at::Tensor& m,
                     at::Tensor& v, at::Tensor& g, float lr, float beta1,
                     float beta2, float eps, float grad_scale, int step,
                     int mode, int bias_correction, float decay);

void apex_fused_adam_cuda(at::Tensor& p, at::Tensor& p_copy, at::Tensor& m,
                          at::Tensor& v, at::Tensor& g, float lr, float beta1,
                          float beta2, float eps, float grad_scale, int step,
                          int mode, int bias_correction, float decay);
