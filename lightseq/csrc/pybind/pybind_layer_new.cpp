#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <string>

#include "context.h"
#include "cuda_util.h"
#include "transformer_encoder_layer.h"
#include "transformer_decoder_layer.h"

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

int create_global_context(bool is_training = true) {
  int context_id;
  if (is_training)
    context_id = Context::create_global_context(StatusType::Training);
  else
    context_id = Context::create_global_context(StatusType::Inference);

#ifdef LIGHTSEQ_cuda
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  Context::global_instance()->set_stream(stream);
#endif
  return context_id;
}

void set_global_context(int context_id) {
  Context::set_global_context(context_id);
}

template <typename T1, typename T2>
int create_transformer_encoder_layer_new(
    int layer_id, int max_batch_tokens, int max_seq_len, int hidden_dim,
    int num_heads, int intermediate_size, float attn_prob_dropout_ratio,
    float activation_dropout_ratio, float hidden_dropout_ratio,
    bool pre_or_postLayerNorm, std::string activation_fn,
    bool mask_future_tokens) {
  auto layer = std::make_shared<TransformerEncoderLayer<T1, T2>>(
      layer_id, max_batch_tokens, max_seq_len, hidden_dim, num_heads,
      intermediate_size, attn_prob_dropout_ratio, activation_dropout_ratio,
      hidden_dropout_ratio, pre_or_postLayerNorm, activation_fn,
      mask_future_tokens);

  Variable *inp(new Variable("input"));
  Variable *inp_mask(new Variable("inp_mask"));

  (*layer)(inp, inp_mask);

  Context::regist_pybind_layer("TransformerEncoderLayer", layer_id, layer);

  layer->before_forward(1, 32);

  std::string T1_dtype = (std::is_same<T1, __half>::value) ? "half" : "float";
  std::string T2_dtype = (std::is_same<T2, __half>::value) ? "half" : "float";

  std::cout << "Encoder layer #" << layer_id << " is created with date type ["
            << T1_dtype << ", " << T2_dtype << "]." << std::endl;

  return 0;
}

template <typename T1, typename T2>
std::vector<torch::Tensor> transformer_encoder_layer_fw(
    int layer_id, const torch::Tensor &input, const torch::Tensor &input_mask,
    bool training_mode) {
  CHECK_INPUT(input);
  CHECK_INPUT(input_mask);

  auto output = torch::empty_like(input);

  const char *input_ptr = (const char *)input.data_ptr();
  const char *input_mask_ptr = (const char *)input_mask.data_ptr();

  char *out_ptr = (char *)output.data_ptr();

  std::shared_ptr<TransformerEncoderLayer<T1, T2>> layer =
      std::static_pointer_cast<TransformerEncoderLayer<T1, T2>>(
          Context::get_pybind_layer("TransformerEncoderLayer", layer_id));

  Variable *inp_node = layer->input(0);
  inp_node->set_value(input_ptr);
  Variable *inp_mask_node = layer->input(1);
  inp_mask_node->set_value(input_mask_ptr);

  Variable *out_node = layer->output(0);
  out_node->set_value(out_ptr);

  layer->before_forward(input.size(0), input.size(1));

  layer->forward();

  return {output};
}

template <typename T1, typename T2>
std::vector<torch::Tensor> transformer_encoder_layer_bw(
    int layer_id, const torch::Tensor &grad_out, const torch::Tensor &output,
    const torch::Tensor &input, const torch::Tensor &input_mask) {
  CHECK_INPUT(grad_out);
  CHECK_INPUT(output);
  CHECK_INPUT(input);
  CHECK_INPUT(input_mask);

  auto grad_inp = torch::empty_like(grad_out);

  // inputs.
  char *grad_output_ptr = (char *)grad_out.data_ptr();
  const char *input_ptr = (const char *)input.data_ptr();
  const char *output_ptr = (const char *)output.data_ptr();
  const char *input_mask_ptr = (const char *)input_mask.data_ptr();

  // outputs.
  char *grad_input_ptr = (char *)grad_inp.data_ptr();

  std::shared_ptr<TransformerEncoderLayer<T1, T2>> layer =
      std::static_pointer_cast<TransformerEncoderLayer<T1, T2>>(
          Context::get_pybind_layer("TransformerEncoderLayer", layer_id));

  Variable *inp_node = layer->input(0);
  inp_node->set_value(input_ptr);
  inp_node->set_grad(grad_input_ptr);
  Variable *inp_mask_node = layer->input(1);
  inp_mask_node->set_value(input_mask_ptr);

  Variable *out_node = layer->output(0);
  out_node->set_value(output_ptr);
  out_node->set_grad(grad_output_ptr);

  layer->backward();

  return {grad_inp};
}

/* Transformer decoder layer */
template <typename T1, typename T2>
int create_transformer_decoder_layer(
    int nshared_layer, int layer_id, int max_batch_tokens, int max_seq_len,
    int hidden_dim, int num_heads, int intermediate_size,
    float attn_prob_dropout_ratio, float activation_dropout_ratio,
    float hidden_dropout_ratio, bool pre_or_postLayerNorm,
    std::string activation_fn) {
  auto layer = std::make_shared<TransformerDecoderLayer<T1, T2>>(
      nshared_layer, layer_id, max_batch_tokens, max_seq_len, hidden_dim,
      num_heads, intermediate_size, attn_prob_dropout_ratio,
      activation_dropout_ratio, hidden_dropout_ratio, pre_or_postLayerNorm,
      activation_fn);

  Context::regist_pybind_layer("TransformerDecoderLayer", layer_id, layer);

  Variable *dec_inp = new Variable("decoder_input");
  Variable *enc_out = new Variable("encoder_out");
  Variable *enc_mask = new Variable("encoder_mask");
  Variable *cache_self_k = new Variable("cache_self_k");
  Variable *cache_self_v = new Variable("cache_self_v");

  (*layer)(dec_inp, enc_out, enc_mask, cache_self_k, cache_self_v);

  if (Context::global_instance()->is_training())
    layer->before_forward(1, 32, 32, -1);
  else
    layer->before_forward(1, 32, 32, 0);

  std::string dtype = (std::is_same<T1, __half>::value) ? "half" : "float";

  std::cout << "Decoder layer #" << layer_id << " is created with date type ["
            << dtype << "]." << std::endl;

  return 0;
}

template <typename T1, typename T2>
std::vector<torch::Tensor> transformer_decoder_layer_fw(
    int layer_id, const torch::Tensor &dec_input,
    const torch::Tensor &enc_output, const torch::Tensor &enc_mask,
    bool training_mode, bool prelayernorm, bool quant_mode,
    std::vector<torch::Tensor> &cache) {
  CHECK_INPUT(dec_input);
  CHECK_INPUT(enc_output);
  CHECK_INPUT(enc_mask);

  const char *dec_input_ptr = (const char *)dec_input.data_ptr();
  const char *enc_output_ptr = (const char *)enc_output.data_ptr();
  const char *enc_mask_ptr = (const char *)enc_mask.data_ptr();

  auto dec_output = torch::empty_like(dec_input);
  char *dec_output_ptr = (char *)dec_output.data_ptr();

  std::shared_ptr<TransformerDecoderLayer<T1, T2>> layer =
      std::static_pointer_cast<TransformerDecoderLayer<T1, T2>>(
          Context::get_pybind_layer("TransformerDecoderLayer", layer_id));

  int batch_size = enc_output.size(0);
  int trg_seq_len = dec_input.size(1);
  int src_seq_len = enc_output.size(1);
  int step = -1;
  std::vector<char *> cache_ptr(5, nullptr);
  if (cache.size() > 0) {
    trg_seq_len = dec_input.size(0) / batch_size;  // beam_size
    step = cache[0].size(2) - 1;
    cache_ptr[0] = (char *)cache[0].data_ptr();  // new dec-self-attn k
    cache_ptr[1] = (char *)cache[1].data_ptr();  // new dec-self-attn v
    if (step > 0) {
      cache_ptr[2] = (char *)cache[2].data_ptr();  // old dec-self-attn k
      cache_ptr[3] = (char *)cache[3].data_ptr();  // old dec-self-attn v
    }
  }

  Variable *inp_node = layer->input(0);
  inp_node->set_value(dec_input_ptr);

  Variable *enc_out_node = layer->input(1);
  enc_out_node->set_value(enc_output_ptr);

  Variable *enc_mask_node = layer->input(2);
  enc_mask_node->set_value(enc_mask_ptr);

  Variable *old_cache_k = layer->input(3);
  old_cache_k->set_value(cache_ptr[2]);

  Variable *old_cache_v = layer->input(4);
  old_cache_v->set_value(cache_ptr[3]);

  Variable *dec_out = layer->output(0);
  dec_out->set_value(dec_output_ptr);

  if (cache.size() > 0) {
    Variable *new_cache_k = layer->output(1);
    new_cache_k->set_value(cache_ptr[0]);
    Variable *new_cache_v = layer->output(2);
    new_cache_v->set_value(cache_ptr[1]);
  }

  layer->before_forward(batch_size, trg_seq_len, src_seq_len, step);

  layer->forward();

  return {dec_output};
}

template <typename T1, typename T2>
void assign_layer_weight_grad(const torch::Tensor &weights,
                              torch::Tensor &grads, std::string layer_name,
                              int layer_id) {
  CHECK_INPUT(weights);
  const T1 *wptr = (const T1 *)weights.data_ptr();

  CHECK_INPUT(grads);
  T2 *gptr = (T2 *)grads.data_ptr();

  if (layer_name == "TransformerEncoderLayer") {
    std::shared_ptr<TransformerEncoderLayer<T1, T2>> layer =
        std::static_pointer_cast<TransformerEncoderLayer<T1, T2>>(
            Context::get_pybind_layer("TransformerEncoderLayer", layer_id));
    layer->load_para_and_grad(wptr, gptr);
  } else if (layer_name == "TransformerDecoderLayer") {
    std::shared_ptr<TransformerDecoderLayer<T1, T2>> layer =
        std::static_pointer_cast<TransformerDecoderLayer<T1, T2>>(
            Context::get_pybind_layer("TransformerDecoderLayer", layer_id));
    layer->load_para_and_grad(wptr, gptr);
  } else {
    printf("Error! layer_name %s is unsupported!\n", layer_name.c_str());
    exit(-1);
  }
  std::cout << layer_name << " #" << layer_id << " bind weights and grads."
            << std::endl;
  return;
}

}  // namespace lightseq

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // create default context
  lightseq::Context::create_global_context(lightseq::StatusType::Training);

  m.def("create_global_context", &lightseq::create_global_context,
        "Create Lightseq Context");
  m.def("set_global_context", &lightseq::set_global_context,
        "Set Lightseq Context");

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

  m.def("transformer_encoder_layer_bw_fp32",
        &lightseq::transformer_encoder_layer_bw<float, float>,
        "LightSeq Transformer Encoder forward with fp32 (CUDA)");
  m.def("transformer_encoder_layer_bw_fp16",
        &lightseq::transformer_encoder_layer_bw<__half, __half>,
        "LightSeq Transformer Encoder forward with fp16 (CUDA)");

  m.def("create_transformer_decoder_layer_new_fp32",
        &lightseq::create_transformer_decoder_layer<float, float>,
        "Create LightSeq Transformer Decoder Layer with fp32 (CUDA)");
  m.def("create_transformer_decoder_layer_new_fp16",
        &lightseq::create_transformer_decoder_layer<__half, __half>,
        "Create LightSeq Transformer Decoder Layer with fp16 (CUDA)");

  m.def("transformer_decoder_layer_fw_fp32",
        &lightseq::transformer_decoder_layer_fw<float, float>,
        "LightSeq Transformer Decoder forward with fp32 (CUDA)");
  m.def("transformer_decoder_layer_fw_fp16",
        &lightseq::transformer_decoder_layer_fw<__half, __half>,
        "LightSeq Transformer Decoder forward with fp16 (CUDA)");

  m.def("assign_layer_weight_grad_fp32",
        &lightseq::assign_layer_weight_grad<float, float>,
        "Bind layer weights and grads");
  m.def("assign_layer_weight_grad_fp16",
        &lightseq::assign_layer_weight_grad<__half, __half>,
        "Bind layer weights and grads");
}
