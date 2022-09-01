#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <string>

#include "context.h"
#include "cuda_util.h"
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

void ContextInitial() {
  static ContextPtr context_ptr;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (context_ptr == nullptr) {
    context_ptr.reset(new Context());
    Context::set_thread_context(context_ptr);
  }
  context_ptr->set_stream(stream);
}

template <typename T1, typename T2>
int create_transformer_encoder_layer_new(
    int layer_id, int max_batch_tokens, int max_seq_len, int hidden_dim,
    int num_heads, int intermediate_size, float attn_prob_dropout_ratio,
    float activation_dropout_ratio, float hidden_dropout_ratio,
    bool pre_or_postLayerNorm, std::string activation_fn,
    bool mask_future_tokens, torch::Tensor &para_ptr, torch::Tensor &grad_ptr) {
  // necessary
  ContextInitial();

  auto layer = std::make_shared<TransformerEncoderLayer<T1, T2>>(
      layer_id, max_batch_tokens, max_seq_len, hidden_dim, num_heads,
      intermediate_size, attn_prob_dropout_ratio, activation_dropout_ratio,
      hidden_dropout_ratio, pre_or_postLayerNorm, activation_fn,
      mask_future_tokens);

  layer->load_para_and_grad(rptr<T1>(para_ptr), rptr<T2>(grad_ptr));

  Variable *inp(new Variable("transformer_encoder_layer_" +
                             std::to_string(layer_id) + "_inp"));
  Variable *inp_mask(new Variable("transformer_encoder_layer_" +
                                  std::to_string(layer_id) + "_inp_mask"));

  Variable *layer_out = (*layer)(inp, inp_mask);

  s_transformer_encoder_layers[layer_id] = layer;

  std::string T1_dtype = (std::is_same<T1, __half>::value) ? "half" : "float";
  std::string T2_dtype = (std::is_same<T2, __half>::value) ? "half" : "float";

  std::cout << "Encoder layer #" << layer_id << " is created with date type ["
            << T1_dtype << ", " << T2_dtype << "]." << std::endl;

  return 0;
}

template <typename T1, typename T2>
void transformer_encoder_layer_fw(int layer_id, torch::Tensor &output,
                                  const torch::Tensor &input,
                                  const torch::Tensor &input_mask,
                                  bool training_mode) {
  CHECK_INPUT(input);
  CHECK_INPUT(input_mask);

  const char *input_ptr = (const char *)input.data_ptr();
  const char *input_mask_ptr = (const char *)input_mask.data_ptr();

  char *out_ptr = (char *)output.data_ptr();

  std::shared_ptr<TransformerEncoderLayer<T1, T2>> layer =
      std::static_pointer_cast<TransformerEncoderLayer<T1, T2>>(
          s_transformer_encoder_layers[layer_id]);

  Variable *inp_node = layer->input(0);
  inp_node->set_value(input_ptr);
  Variable *inp_mask_node = layer->input(1);
  inp_mask_node->set_value(input_mask_ptr);

  Variable *out_node = layer->output(0);
  out_node->set_value(out_ptr);

  layer->before_forward(input.size(0), input.size(1));

  layer->forward();

  return;
}

}  // namespace lightseq

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("create_transformer_encoder_layer_new_fp32",
        &lightseq::create_transformer_encoder_layer_new<float, float>,
        "Create LightSeq Transformer Encoder Layer with fp32 (CUDA)");
  m.def("create_transformer_encoder_layer_new_fp16",
        &lightseq::create_transformer_encoder_layer_new<__half, __half>,
        "Create LightSeq Transformer Encoder Layer with fp16 (CUDA)");

  m.def("transformer_encoder_layer_fw_fp32",
        &lightseq::transformer_encoder_layer_fw<float, float>,
        "LightSeq Transformer Encoder forward with fp32 (CUDA)");
  m.def("transformer_encoder_layer_fw_fp16",
        &lightseq::transformer_encoder_layer_fw<__half, __half>,
        "LightSeq Transformer Encoder forward with fp16 (CUDA)");
}
