#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <string>

#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/uuid/uuid_io.hpp>         // streaming operators etc.

#include "context.h"
#include "transformer_encoder_layer.h"


// x is torch::Tensor
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

namespace lightseq {
  
template <typename T>
const T *rptr(const torch::Tensor &tensor) {
  return reinterpret_cast<const T *>(tensor.data_ptr());
}

template <typename T>
T *rptr(torch::Tensor &tensor) {
  return reinterpret_cast<T *>(tensor.data_ptr());
}

static std::unordered_map<int, std::shared_ptr<void>>
    s_transformer_encoder_layers;
static std::unordered_map<int, std::shared_ptr<void>> s_cross_entropy_layers;

static ContextPtr global_context_ptr = Context();

template <typename T1, typename T2>
int create_transformer_encoder_layer(
    int layer_id, int max_batch_tokens, int max_seq_len, int hidden_dim,
    int num_heads, int intermediate_size, float attn_prob_dropout_ratio,
    float activation_dropout_ratio, float hidden_dropout_ratio,
    bool pre_or_postLayerNorm, std::string activation_fn,
    bool mask_future_tokens, const torch::Tensor& para_ptr, torch::Tensor& grad_ptr) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  Context::Instance().set_stream(stream);
  
  int layer_offset = 0;

  auto layer = std::make_shared<TransformerEncoderLayer<T>>(
      layer_id, max_batch_tokens, max_seq_len, hidden_dim, num_heads,
      intermediate_size, attn_prob_dropout_ratio, activation_dropout_ratio,
      hidden_dropout_ratio, pre_or_postLayerNorm, activation_fn,
      mask_future_tokens, rptr(para_ptr), rptr(grad_ptr), layer_offset);
    
  Variable* inp(new Variable("transformer_encoder_layer_" + std::to_string(layer_id) + "_inp", 
                            max_batch_tokens * hidden_dim * sizeof(T1), 
                            max_batch_tokens * hidden_dim * sizeof(T2)));
  Variable* inp_mask(new Variable("transformer_encoder_layer_" + std::to_string(layer_id) + "_inp_mask", 
                            max_batch_tokens * hidden_dim * sizeof(T1), 
                            max_batch_tokens * hidden_dim * sizeof(T2)));

  Variable* layer_out = (*layer)(inp, inp_mask);

  s_transformer_encoder_layers[layer_id] = layer;

  std::string T1_dtype = (std::is_same<T1, __half>::value) ? "half" : "float";
  std::string T2_dtype = (std::is_same<T2, __half>::value) ? "half" : "float";

  std::cout << "Encoder layer #" << layer_id << " is created with date type ["
            << T1_dtype << ", " << T2_dtype << "]." << std::endl;

  return 0;
}

template <typename T1, typename T2>
std::vector<torch::Tensor> transformer_encoder_fw(
    int layer_id, torch::Tensor& output, const torch::Tensor &input, 
    const torch::Tensor &input_mask, bool training_mode) {

  CHECK_INPUT(input);
  CHECK_INPUT(input_mask);

  const T1 *input_ptr = (const T1*)input.data_ptr();
  const T1 *input_mask_ptr = (const T1*)input_mask.data_ptr();

  T1 *out_ptr = (T1 *)output.data_ptr();

  std::shared_ptr<TransformerEncoderLayer<T>> layer =
      std::static_pointer_cast<TransformerEncoderLayer<T>>(
          s_transformer_encoder_layers[layer_id]);

  
}

} // namespace lightseq



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("transformer_encoder_layer_fw_fp32",
        &transformer_encoder_layer_fw<float>,
        "LightSeq Transformer Encoder forward with fp32 (CUDA)");
  
  m.def("transformer_encoder_layer_fw_fp16",
        &transformer_encoder_layer_fw<__half>,
        "LightSeq Transformer Encoder forward with fp16 (CUDA)");
}
