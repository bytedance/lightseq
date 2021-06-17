#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

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

static std::unordered_map<int, std::shared_ptr<void>>
    s_transformer_encoder_layers;
static std::unordered_map<int, std::shared_ptr<void>> s_cross_entropy_layers;

template <typename T>
int create_transformer_encoder_layer(int layer_id, int max_batch_tokens,
                                     int max_seq_len, int hidden_dim,
                                     int num_heads, int intermediate_size,
                                     float attn_prob_dropout_ratio,
                                     float activation_dropout_ratio,
                                     float hidden_dropout_ratio,
                                     bool pre_or_postLayerNorm) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  Context::Instance().set_stream(stream);
  auto layer = std::make_shared<TransformerEncoderLayer<T>>(
      layer_id, max_batch_tokens, max_seq_len, hidden_dim, num_heads,
      intermediate_size, attn_prob_dropout_ratio, activation_dropout_ratio,
      hidden_dropout_ratio, pre_or_postLayerNorm);

  s_transformer_encoder_layers[layer_id] = layer;

  std::string dtype = (std::is_same<T, __half>::value) ? "half" : "float";

  std::cout << "Encoder layer #" << layer_id << " is created with date type ["
            << dtype << "]." << std::endl;

  return 0;
}

template <typename T>
std::vector<torch::Tensor> transformer_encoder_layer_fw(
    int layer_id, const torch::Tensor &input, const torch::Tensor &input_mask,
    bool training_mode, bool prelayernorm) {
  CHECK_INPUT(input);
  CHECK_INPUT(input_mask);

  const T *input_ptr = (const T *)input.data_ptr();
  const T *input_mask_ptr = (const T *)input_mask.data_ptr();

  auto output = torch::empty_like(input);
  T *out_ptr = (T *)output.data_ptr();

  std::shared_ptr<TransformerEncoderLayer<T>> layer =
      std::static_pointer_cast<TransformerEncoderLayer<T>>(
          s_transformer_encoder_layers[layer_id]);
  layer->set_cur_batch_shape(input.size(0), input.size(1));
  layer->SetTrainingMode(training_mode);
  layer->Forward(input_ptr, input_mask_ptr, out_ptr);

  return {output};
}

template <typename T>
std::vector<torch::Tensor> transformer_encoder_layer_bw(
    int layer_id, const torch::Tensor &grad_dec_output,
    const torch::Tensor &output, const torch::Tensor &input,
    const torch::Tensor &input_mask) {
  auto g_output = grad_dec_output.contiguous();
  CHECK_INPUT(g_output);
  CHECK_INPUT(output);
  CHECK_INPUT(input);
  CHECK_INPUT(input_mask);

  auto grad_input = torch::empty_like(input);

  // inputs.
  const T *grad_dec_output_ptr = (const T *)g_output.data_ptr();
  const T *input_ptr = (const T *)input.data_ptr();
  const T *output_ptr = (const T *)output.data_ptr();
  const T *input_mask_ptr = (const T *)input_mask.data_ptr();

  // outputs.
  T *grad_input_ptr = (T *)grad_input.data_ptr();

  std::shared_ptr<TransformerEncoderLayer<T>> layer =
      std::static_pointer_cast<TransformerEncoderLayer<T>>(
          s_transformer_encoder_layers[layer_id]);
  layer->set_cur_batch_shape(g_output.size(0), g_output.size(1));
  layer->Backward(grad_dec_output_ptr, input_ptr, output_ptr, input_mask_ptr,
                  grad_input_ptr);

  return {grad_input};
}

static std::unordered_map<int, std::shared_ptr<void>>
    s_transformer_decoder_layers;

template <typename T>
int create_transformer_decoder_layer(int layer_id, int max_batch_tokens,
                                     int max_seq_len, int hidden_dim,
                                     int num_heads, int intermediate_size,
                                     float attn_prob_dropout_ratio,
                                     float activation_dropout_ratio,
                                     float hidden_dropout_ratio,
                                     bool pre_or_postLayerNorm) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  Context::Instance().set_stream(stream);
  auto layer = std::make_shared<TransformerDecoderLayer<T>>(
      layer_id, max_batch_tokens, max_seq_len, hidden_dim, num_heads,
      intermediate_size, attn_prob_dropout_ratio, activation_dropout_ratio,
      hidden_dropout_ratio, pre_or_postLayerNorm);

  s_transformer_decoder_layers[layer_id] = layer;

  std::string dtype = (std::is_same<T, __half>::value) ? "half" : "float";

  std::cout << "Decoder layer #" << layer_id << " is created with date type ["
            << dtype << "]." << std::endl;

  return 0;
}

template <typename T>
std::vector<torch::Tensor> transformer_decoder_layer_fw(
    int layer_id, const torch::Tensor &dec_input,
    const torch::Tensor &enc_output, const torch::Tensor &enc_mask,
    bool training_mode, bool prelayernorm, std::vector<torch::Tensor> &cache) {
  CHECK_INPUT(dec_input);
  CHECK_INPUT(enc_output);
  CHECK_INPUT(enc_mask);

  const T *dec_input_ptr = (const T *)dec_input.data_ptr();
  const T *enc_output_ptr = (const T *)enc_output.data_ptr();
  const T *enc_mask_ptr = (const T *)enc_mask.data_ptr();

  auto dec_output = torch::empty_like(dec_input);
  T *dec_output_ptr = (T *)dec_output.data_ptr();

  std::shared_ptr<TransformerDecoderLayer<T>> layer =
      std::static_pointer_cast<TransformerDecoderLayer<T>>(
          s_transformer_decoder_layers[layer_id]);

  int batch_size = enc_output.size(0);
  int trg_seq_len = dec_input.size(1);
  int src_seq_len = enc_output.size(1);
  int step = -1;
  std::vector<T *> cache_ptr;
  if (cache.size() > 0) {
    trg_seq_len = dec_input.size(0) / batch_size;  // beam_size
    step = cache[0].size(2) - 1;
    cache_ptr = {nullptr, nullptr, nullptr, nullptr, nullptr};
    cache_ptr[0] = (T *)cache[0].data_ptr();  // new dec-self-attn k
    cache_ptr[1] = (T *)cache[1].data_ptr();  // new dec-self-attn v
    if (step > 0) {
      cache_ptr[2] = (T *)cache[2].data_ptr();  // old dec-self-attn k
      cache_ptr[3] = (T *)cache[3].data_ptr();  // old dec-self-attn v
    }
    if (step == 0 && layer_id == 0) {
      cache_ptr[4] =
          (T *)cache[cache.size() - 1].data_ptr();  // enc-dec-attn kv
    }
  }
  layer->set_cur_batch_shape(batch_size, trg_seq_len, src_seq_len, step);
  layer->SetTrainingMode(training_mode);
  layer->Forward(dec_input_ptr, enc_output_ptr, enc_mask_ptr, dec_output_ptr,
                 cache_ptr);

  return {dec_output};
}

template <typename T>
std::vector<torch::Tensor> transformer_decoder_layer_bw(
    int layer_id, const torch::Tensor &grad_dec_output,
    const torch::Tensor &dec_output, const torch::Tensor &dec_input,
    const torch::Tensor &enc_output, const torch::Tensor &enc_mask) {
  auto g_dec_output = grad_dec_output.contiguous();
  CHECK_INPUT(g_dec_output);
  CHECK_INPUT(dec_output);
  CHECK_INPUT(dec_input);
  CHECK_INPUT(enc_output);
  CHECK_INPUT(enc_mask);

  const T *grad_dec_output_ptr = (const T *)g_dec_output.data_ptr();
  const T *dec_input_ptr = (const T *)dec_input.data_ptr();
  const T *dec_output_ptr = (const T *)dec_output.data_ptr();
  const T *enc_output_ptr = (const T *)enc_output.data_ptr();
  const T *enc_mask_ptr = (const T *)enc_mask.data_ptr();

  auto grad_dec_input = torch::empty_like(dec_input);
  T *grad_dec_input_ptr = (T *)grad_dec_input.data_ptr();

  std::shared_ptr<TransformerDecoderLayer<T>> layer =
      std::static_pointer_cast<TransformerDecoderLayer<T>>(
          s_transformer_decoder_layers[layer_id]);

  T *grad_enc_output_ptr = nullptr;
  auto grad_enc_output = torch::empty_like(enc_output);
  if (layer_id == 0) {
    grad_enc_output_ptr = (T *)grad_enc_output.data_ptr();
  }
  layer->set_cur_batch_shape(g_dec_output.size(0), g_dec_output.size(1),
                             enc_output.size(1));
  layer->Backward(grad_dec_output_ptr, dec_input_ptr, enc_output_ptr,
                  enc_mask_ptr, dec_output_ptr, grad_dec_input_ptr,
                  grad_enc_output_ptr);
  if (layer_id == 0) {
    return {grad_dec_input, grad_enc_output};
  } else {
    return {grad_dec_input};
  }
}

static std::unordered_map<int, std::shared_ptr<void>>
    s_transformer_embedding_layers;

template <typename T>
int create_transformer_embedding_layer(int layer_id,
                                       const torch::Tensor &pos_embeddings,
                                       int max_batch_tokens, int embedding_dim,
                                       int vocab_size, float dropout_ratio,
                                       int padding_idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  Context::Instance().set_stream(stream);
  const T *pos_embeddings_ptr = (const T *)pos_embeddings.data_ptr();

  auto layer = std::make_shared<TransformerEmbeddingLayer<T>>(
      layer_id, pos_embeddings_ptr, max_batch_tokens, embedding_dim, vocab_size,
      dropout_ratio, padding_idx);

  s_transformer_embedding_layers[layer_id] = layer;

  std::string dtype = (std::is_same<T, __half>::value) ? "half" : "float";

  std::cout << "Embedding layer #" << layer_id << " is created with date type ["
            << dtype << "]." << std::endl;

  return 0;
}

template <typename T>
std::vector<torch::Tensor> transformer_embedding_layer_fw(
    int layer_id, const torch::Tensor &input, int step, bool training_mode) {
  CHECK_INPUT(input);
  const int *input_ptr = (const int *)input.data_ptr();

  std::shared_ptr<TransformerEmbeddingLayer<T>> layer =
      std::static_pointer_cast<TransformerEmbeddingLayer<T>>(
          s_transformer_embedding_layers[layer_id]);

  auto dtype =
      (std::is_same<T, __half>::value) ? torch::kFloat16 : torch::kFloat32;

  auto options = torch::TensorOptions()
                     .dtype(dtype)
                     .layout(torch::kStrided)
                     .device(torch::kCUDA)
                     .requires_grad(true);
  auto output = torch::empty(
      {input.size(0), input.size(1), layer->EmbeddingDim()}, options);
  T *out_ptr = (T *)output.data_ptr();

  layer->set_cur_batch_shape(input.size(0), input.size(1));
  layer->SetTrainingMode(training_mode);
  layer->Forward(input_ptr, out_ptr, step);

  return {output};
}

template <typename T>
void transformer_embedding_layer_bw(int layer_id,
                                    const torch::Tensor &grad_output,
                                    const torch::Tensor &input) {
  auto g_output = grad_output.contiguous();
  CHECK_INPUT(g_output);
  CHECK_INPUT(input);

  const T *grad_output_ptr = (const T *)g_output.data_ptr();
  const int *input_ptr = (const int *)input.data_ptr();

  std::shared_ptr<TransformerEmbeddingLayer<T>> layer =
      std::static_pointer_cast<TransformerEmbeddingLayer<T>>(
          s_transformer_embedding_layers[layer_id]);

  layer->set_cur_batch_shape(g_output.size(0), g_output.size(1));
  layer->Backward(grad_output_ptr, input_ptr);
  return;
}

template <typename T>
void assign_layer_weight_grad(const torch::Tensor &weights,
                              torch::Tensor &grads, std::string layer_name,
                              int layer_id) {
  CHECK_INPUT(weights);
  const T *wptr = (const T *)weights.data_ptr();

  CHECK_INPUT(grads);
  T *gptr = (T *)grads.data_ptr();

  if (layer_name == "TransformerDecoderLayer") {
    std::shared_ptr<TransformerDecoderLayer<T>> layer =
        std::static_pointer_cast<TransformerDecoderLayer<T>>(
            s_transformer_decoder_layers[layer_id]);
    layer->assign_weight_ptr(wptr);
    layer->assign_grad_ptr(gptr);
  } else if (layer_name == "TransformerEncoderLayer") {
    std::shared_ptr<TransformerEncoderLayer<T>> layer =
        std::static_pointer_cast<TransformerEncoderLayer<T>>(
            s_transformer_encoder_layers[layer_id]);
    layer->assign_weight_ptr(wptr);
    layer->assign_grad_ptr(gptr);
  } else if (layer_name == "TransformerEmbeddingLayer") {
    std::shared_ptr<TransformerEmbeddingLayer<T>> layer =
        std::static_pointer_cast<TransformerEmbeddingLayer<T>>(
            s_transformer_embedding_layers[layer_id]);
    layer->assign_weight_ptr(wptr);
    layer->assign_grad_ptr(gptr);
  }
  std::cout << layer_name << " #" << layer_id << " bind weights and grads."
            << std::endl;
  return;
}

template <typename T>
int create_cross_entropy_layer(const int layer_id, const float epsilon,
                               const int padding_idx,
                               const int max_batch_tokens) {
  auto layer = std::make_shared<CrossEntropyLayer<T>>(epsilon, padding_idx,
                                                      max_batch_tokens);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  Context::Instance().set_stream(stream);
  s_cross_entropy_layers[layer_id] = layer;

  std::string dtype = (std::is_same<T, __half>::value) ? "half" : "float";

  std::cout << "CrossEntropyLayer is created with date type [" << dtype << "]."
            << std::endl;

  return 0;
}

template <typename T>
std::vector<torch::Tensor> cross_entropy_layer_fw(
    const int layer_id, const torch::Tensor &inputs,
    const torch::Tensor &targets) {
  CHECK_INPUT(inputs);
  CHECK_INPUT(targets);
  AT_ASSERTM(targets.dtype() == torch::kInt32, "targets must be int32");

  const T *inputs_ptr = static_cast<const T *>(inputs.data_ptr());
  const int *targets_ptr = static_cast<const int *>(targets.data_ptr());

  int batch_size = inputs.size(0);
  int seq_len = inputs.size(1);
  int vocab_size = inputs.size(2);

  std::shared_ptr<CrossEntropyLayer<T>> layer =
      std::static_pointer_cast<CrossEntropyLayer<T>>(
          s_cross_entropy_layers[layer_id]);

  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat32)
                     .layout(torch::kStrided)
                     .device(torch::kCUDA, inputs.device().index());
  auto outputs = torch::zeros({1}, options);
  auto nll_loss = torch::zeros({1}, options);
  float *outputs_ptr = static_cast<float *>(outputs.data_ptr());
  float *nll_loss_ptr = static_cast<float *>(nll_loss.data_ptr());

  layer->set_cur_batch_shape(batch_size, seq_len, vocab_size);
  layer->Forward(inputs_ptr, targets_ptr, outputs_ptr, nll_loss_ptr);
  return {outputs, nll_loss};
}

template <typename T>
std::vector<torch::Tensor> cross_entropy_layer_bw(
    const int layer_id, const torch::Tensor &grad_outputs,
    const torch::Tensor &inputs, const torch::Tensor &targets) {
  CHECK_INPUT(grad_outputs);
  CHECK_INPUT(inputs);
  CHECK_INPUT(targets);
  AT_ASSERTM(targets.dtype() == torch::kInt32, "targets must be int32");

  const float *grad_outputs_ptr =
      static_cast<const float *>(grad_outputs.data_ptr());
  const T *inputs_ptr = static_cast<const T *>(inputs.data_ptr());
  const int *targets_ptr = static_cast<const int *>(targets.data_ptr());

  int batch_size = inputs.size(0);
  int seq_len = inputs.size(1);
  int vocab_size = inputs.size(2);

  auto grad_inputs = torch::zeros_like(inputs);
  T *grad_inputs_ptr = static_cast<T *>(grad_inputs.data_ptr());

  std::shared_ptr<CrossEntropyLayer<T>> layer =
      std::static_pointer_cast<CrossEntropyLayer<T>>(
          s_cross_entropy_layers[layer_id]);

  layer->set_cur_batch_shape(batch_size, seq_len, vocab_size);
  layer->Backward(grad_outputs_ptr, inputs_ptr, targets_ptr, grad_inputs_ptr);
  return {grad_inputs};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("transformer_encoder_layer_fw_fp32",
        &transformer_encoder_layer_fw<float>,
        "LightSeq Transformer Encoder forward with fp32 (CUDA)");
  m.def("transformer_encoder_layer_fw_fp16",
        &transformer_encoder_layer_fw<__half>,
        "LightSeq Transformer Encoder forward with fp16 (CUDA)");
  m.def("transformer_encoder_layer_bw_fp32",
        &transformer_encoder_layer_bw<float>,
        "LightSeq Transformer Encoder backward with fp32 (CUDA)");
  m.def("transformer_encoder_layer_bw_fp16",
        &transformer_encoder_layer_bw<__half>,
        "LightSeq Transformer Encoder backward with fp16 (CUDA)");
  m.def("create_transformer_encoder_layer_fp32",
        &create_transformer_encoder_layer<float>,
        "Create LightSeq Transformer Encoder Layer with fp32 (CUDA)");
  m.def("create_transformer_encoder_layer_fp16",
        &create_transformer_encoder_layer<__half>,
        "Create LightSeq Transformer Encoder Layer with fp16 (CUDA)");
  m.def("transformer_decoder_layer_fw_fp32",
        &transformer_decoder_layer_fw<float>,
        "LightSeq Transformer Decoder forward with fp32 (CUDA)");
  m.def("transformer_decoder_layer_fw_fp16",
        &transformer_decoder_layer_fw<__half>,
        "LightSeq Transformer Decoder forward with fp16 (CUDA)");
  m.def("transformer_decoder_layer_bw_fp32",
        &transformer_decoder_layer_bw<float>,
        "LightSeq Transformer Decoder backward with fp32 (CUDA)");
  m.def("transformer_decoder_layer_bw_fp16",
        &transformer_decoder_layer_bw<__half>,
        "LightSeq Transformer Decoder backward with fp16 (CUDA)");
  m.def("create_transformer_decoder_layer_fp32",
        &create_transformer_decoder_layer<float>,
        "Create LightSeq Transformer Decoder Layer with fp32 (CUDA)");
  m.def("create_transformer_decoder_layer_fp16",
        &create_transformer_decoder_layer<__half>,
        "Create LightSeq Transformer Decoder Layer with fp16 (CUDA)");
  m.def("transformer_embedding_layer_fw_fp32",
        &transformer_embedding_layer_fw<float>,
        "LightSeq Transformer Embedding forward with fp32 (CUDA)");
  m.def("transformer_embedding_layer_fw_fp16",
        &transformer_embedding_layer_fw<__half>,
        "LightSeq Transformer Embedding forward with fp16 (CUDA)");
  m.def("transformer_embedding_layer_bw_fp32",
        &transformer_embedding_layer_bw<float>,
        "LightSeq Transformer Embedding backward with fp32 (CUDA)");
  m.def("transformer_embedding_layer_bw_fp16",
        &transformer_embedding_layer_bw<__half>,
        "LightSeq Transformer Embedding backward with fp16 (CUDA)");
  m.def("create_transformer_embedding_layer_fp32",
        &create_transformer_embedding_layer<float>,
        "Create LightSeq Transformer Embedding Layer with fp32 (CUDA)");
  m.def("create_transformer_embedding_layer_fp16",
        &create_transformer_embedding_layer<__half>,
        "Create LightSeq Transformer Embedding Layer with fp16 (CUDA)");
  m.def("create_cross_entropy_layer_fp32", &create_cross_entropy_layer<float>,
        "Create LightSeq Cross Entropy Layer with fp32 (CUDA)");
  m.def("create_cross_entropy_layer_fp16", &create_cross_entropy_layer<__half>,
        "Create LightSeq Cross Entropy Layer with fp16 (CUDA)");
  m.def("cross_entropy_layer_fw_fp32", &cross_entropy_layer_fw<float>,
        "LightSeq Cross Entropy forward with fp32 (CUDA)");
  m.def("cross_entropy_layer_fw_fp16", &cross_entropy_layer_fw<__half>,
        "LightSeq Cross Entropy forward with fp16 (CUDA)");
  m.def("cross_entropy_layer_bw_fp32", &cross_entropy_layer_bw<float>,
        "LightSeq Cross Entropy backward with fp32 (CUDA)");
  m.def("cross_entropy_layer_bw_fp16", &cross_entropy_layer_bw<__half>,
        "LightSeq Cross Entropy backward with fp16 (CUDA)");
  m.def("assign_layer_weight_grad_fp32", &assign_layer_weight_grad<float>,
        "Bind layer weights and grads");
  m.def("assign_layer_weight_grad_fp16", &assign_layer_weight_grad<__half>,
        "Bind layer weights and grads");
}
