#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <string>

#include "context.h"
#include "cross_entropy_layer.h"
#include "transformer_decoder_layer.h"
#include "transformer_embedding_layer.h"
#include "transformer_encoder_layer.h"

// x is torch::Tensor
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

template <typename T1, typename T2>


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    
}
