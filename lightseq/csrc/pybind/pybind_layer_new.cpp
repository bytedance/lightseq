#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <string>

#include "context.h"
#include "cuda_util.h"
#include "transformer_encoder_layer.h"
#include "transformer_decoder_layer.h"
#include "sdpa_layer.h"

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
    float activation_dropout_ratio, float hidden_dropout_ratio, bool is_pre_ln,
    std::string activation_fn, bool mask_future_tokens) {
  auto layer = std::make_shared<TransformerEncoderLayer<T1, T2>>(
      layer_id, max_batch_tokens, max_seq_len, hidden_dim, num_heads,
      intermediate_size, attn_prob_dropout_ratio, activation_dropout_ratio,
      hidden_dropout_ratio, is_pre_ln, activation_fn, mask_future_tokens);

  Variable *inp(new Variable("input", g_dtype<T1>(), g_dtype<T2>()));
  Variable *inp_mask(new Variable("inp_mask", g_dtype<T1>()));

  (*layer)(inp, inp_mask);

  Context::regist_pybind_layer("TransformerEncoderLayer", layer_id, layer);

  std::string T1_dtype = (std::is_same<T1, __half>::value) ? "half" : "float";
  std::string T2_dtype = (std::is_same<T2, __half>::value) ? "half" : "float";

  std::cout << "Encoder layer #" << layer_id << " is created with date type ["
            << T1_dtype << ", " << T2_dtype << "]." << std::endl;

  return 0;
}

template <typename T1, typename T2>
std::vector<torch::Tensor> transformer_encoder_layer_fw(
    int layer_id, const torch::Tensor &input, const torch::Tensor &input_mask) {
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
  inp_node->set_shape(
      {size_t(input.size(0)), size_t(input.size(1)), size_t(input.size(2))});
  Variable *inp_mask_node = layer->input(1);
  inp_mask_node->set_value(input_mask_ptr);
  inp_mask_node->set_shape(
      {size_t(input_mask.size(0)), size_t(input_mask.size(1))});

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
    float hidden_dropout_ratio, bool is_pre_ln, std::string activation_fn) {
  auto layer = std::make_shared<TransformerDecoderLayer<T1, T2>>(
      nshared_layer, layer_id, max_batch_tokens, max_seq_len, hidden_dim,
      num_heads, intermediate_size, attn_prob_dropout_ratio,
      activation_dropout_ratio, hidden_dropout_ratio, is_pre_ln, activation_fn,
      true);

  std::shared_ptr<Context> context_ptr = Context::global_instance();
  Variable *total_enc_kv = nullptr;
  if (layer_id == 0) {
    Variable *enc_out =
        new Variable("encoder_out", g_dtype<T1>(), g_dtype<T2>());
    std::shared_ptr<EncDecKvLayer<T1, T2>> total_enc_kv_layer(
        new EncDecKvLayer<T1, T2>(nshared_layer, max_batch_tokens, hidden_dim,
                                  num_heads));

    Context::regist_pybind_layer("EncDecKvLayer", 0, total_enc_kv_layer);

    total_enc_kv = (*total_enc_kv_layer)(enc_out);
    context_ptr->register_object("EncDecKvOut", total_enc_kv);
  } else {
    total_enc_kv =
        static_cast<Variable *>(context_ptr->get_object("EncDecKvOut"));
  }

  Variable *dec_inp =
      new Variable("decoder_input", g_dtype<T1>(), g_dtype<T2>());
  Variable *enc_mask = new Variable("encoder_mask", g_dtype<T1>());
  Variable *cache_self_k =
      new Variable("cache_self_k", g_dtype<T1>(), g_dtype<T2>());
  Variable *cache_self_v =
      new Variable("cache_self_v", g_dtype<T1>(), g_dtype<T2>());

  (*layer)(dec_inp, total_enc_kv, enc_mask, cache_self_k, cache_self_v);

  std::string dtype = (std::is_same<T1, __half>::value) ? "half" : "float";

  std::cout << "Decoder layer #" << layer_id << " is created with date type ["
            << dtype << "]." << std::endl;

  Context::regist_pybind_layer("TransformerDecoderLayer", layer_id, layer);
  return 0;
}

template <typename T1, typename T2>
std::vector<torch::Tensor> transformer_decoder_layer_fw(
    int layer_id, const torch::Tensor &dec_input,
    const torch::Tensor &enc_output, const torch::Tensor &enc_mask,
    bool prelayernorm, bool quant_mode, std::vector<torch::Tensor> &cache,
    int cur_step = -1) {
  CHECK_INPUT(dec_input);
  CHECK_INPUT(enc_output);
  CHECK_INPUT(enc_mask);

  if (cache.size() != 4) {
    std::string error_message =
        "Error Occurred! input cache is invalid format.\n";
    throw std::runtime_error(error_message);
  }

  const char *dec_input_ptr = (const char *)dec_input.data_ptr();
  const char *enc_output_ptr = (const char *)enc_output.data_ptr();
  const char *enc_mask_ptr = (const char *)enc_mask.data_ptr();

  auto dec_output = torch::empty_like(dec_input);
  char *dec_output_ptr = (char *)dec_output.data_ptr();

  std::shared_ptr<TransformerDecoderLayer<T1, T2>> layer =
      std::static_pointer_cast<TransformerDecoderLayer<T1, T2>>(
          Context::get_pybind_layer("TransformerDecoderLayer", layer_id));

  size_t batch_size = enc_output.size(0);
  // size_t batch_beam = dec_input.size(0);
  size_t trg_seq_len = dec_input.size(1);
  size_t src_seq_len = enc_output.size(1);
  size_t hidden_size = dec_input.size(2);

  std::shared_ptr<Context> _context_ptr = layer->get_context();

  if (layer_id == 0) {
    std::shared_ptr<EncDecKvLayer<T1, T2>> enc_kv_layer =
        std::static_pointer_cast<EncDecKvLayer<T1, T2>>(
            Context::get_pybind_layer("EncDecKvLayer", 0));
    Variable *enc_out_node = enc_kv_layer->input(0);
    enc_out_node->set_value(enc_output_ptr);
    enc_out_node->set_shape({batch_size, src_seq_len, hidden_size});

    enc_kv_layer->before_forward(batch_size, src_seq_len);
    enc_kv_layer->forward();
  }

  Variable *inp_node = layer->input(0);
  inp_node->set_value(dec_input_ptr);
  inp_node->set_shape({batch_size, trg_seq_len, hidden_size});

  Variable *enc_mask_node = layer->input(2);
  enc_mask_node->set_value(enc_mask_ptr);
  enc_mask_node->set_shape({batch_size, src_seq_len});

  Variable *cache_k = layer->input(3);
  Variable *cache_v = layer->input(4);
  if (_context_ptr->is_inference()) {
    cache_k->set_value((char *)cache[2].data_ptr());
    cache_k->set_shape({batch_size, size_t(cur_step), hidden_size});

    cache_v->set_value((char *)cache[3].data_ptr());
    cache_v->set_shape({batch_size, size_t(cur_step), hidden_size});
  }

  Variable *dec_out = layer->output(0);
  dec_out->set_value(dec_output_ptr);
  dec_out->set_shape({batch_size, trg_seq_len, hidden_size});

  Variable *new_cache_k = layer->output(1);
  new_cache_k->set_value((char *)cache[0].data_ptr());
  new_cache_k->set_shape({batch_size, size_t(cur_step + 1), hidden_size});

  Variable *new_cache_v = layer->output(2);
  new_cache_v->set_value((char *)cache[1].data_ptr());
  new_cache_v->set_shape({batch_size, size_t(cur_step + 1), hidden_size});

  layer->before_forward(batch_size, trg_seq_len, src_seq_len, cur_step);

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
    size_t offset = layer->load_para_and_grad(wptr, gptr);
    if (layer_id == 0) {
      std::shared_ptr<EncDecKvLayer<T1, T2>> enc_kv_layer =
          std::static_pointer_cast<EncDecKvLayer<T1, T2>>(
              Context::get_pybind_layer("EncDecKvLayer", 0));
      enc_kv_layer->load_para_and_grad(wptr + offset, gptr + offset);
    }
  } else {
    printf("Error! layer_name %s is unsupported!\n", layer_name.c_str());
    exit(-1);
  }
  std::cout << layer_name << " #" << layer_id << " bind weights and grads."
            << std::endl;
  return;
}

template <typename T1, typename T2>
int create_sdpa_layer(int layer_id, int max_batch_tokens, int max_seq_len,
                      int head_dim, int num_heads,
                      float attn_prob_dropout_ratio) {
  Variable *q_var = new Variable("query", g_dtype<T1>());
  Variable *k_var = new Variable("key", g_dtype<T1>());
  Variable *v_var = new Variable("value", g_dtype<T1>());
  Variable *mask_var = nullptr;  // FIXME later, only cover non mask

  std::shared_ptr<SDPALayer<T1, T2>> sdpal =
      std::make_shared<SDPALayer<T1, T2>>(max_batch_tokens, max_seq_len,
                                          head_dim, num_heads,
                                          attn_prob_dropout_ratio);
  Variable *res_var = (*sdpal)(q_var, k_var, v_var, mask_var);

  Context::regist_pybind_layer("SDPALayer", layer_id, sdpal);

  return 0;
}

template <typename T1, typename T2>
std::vector<torch::Tensor> sdpa_layer_fw(
    int layer_id, const torch::Tensor &query, const torch::Tensor &key,
    const torch::Tensor &value, const torch::Tensor &mask, int batch_size,
    int query_len, int kv_len, int kv_size, bool mask_future) {
  CHECK_INPUT(query);
  CHECK_INPUT(key);
  CHECK_INPUT(value);

  auto result = torch::empty_like(query);

  const char *query_ptr = (const char *)query.data_ptr();
  const char *key_ptr = (const char *)key.data_ptr();
  const char *value_ptr = (const char *)value.data_ptr();

  char *res_ptr = (char *)result.data_ptr();

  std::shared_ptr<SDPALayer<T1, T2>> layer =
      std::static_pointer_cast<SDPALayer<T1, T2>>(
          Context::get_pybind_layer("SDPALayer", layer_id));

  Variable *query_node = layer->input(0);
  query_node->set_value(query_ptr);
  query_node->set_shape(
      {size_t(query.size(0)), size_t(query.size(1)), size_t(query.size(2))});

  Variable *key_node = layer->input(1);
  key_node->set_value(key_ptr);
  key_node->set_shape(
      {size_t(key.size(0)), size_t(key.size(1)), size_t(key.size(2))});

  Variable *value_node = layer->input(1);
  value_node->set_value(value_ptr);
  value_node->set_shape(
      {size_t(value.size(0)), size_t(value.size(1)), size_t(value.size(2))});

  Variable *res_node = layer->output(0);
  res_node->set_value(res_ptr);

  layer->before_forward(batch_size, query_len, kv_len, kv_size, mask_future);

  layer->forward();

  return {result};
}

template <typename T1, typename T2>
void torch_sdpa_layer(const torch::Tensor &query, const torch::Tensor &key,
                      const torch::Tensor &value, const torch::Tensor &mask,
                      torch::Tensor &res, int max_batch_tokens, int max_seq_len,
                      int head_dim, int num_heads,
                      float attn_prob_dropout_ratio, int batch_size,
                      int query_len, int kv_len, int kv_size,
                      bool mask_future) {
  Context::create_global_context(StatusType::Inference);
  std::shared_ptr<Context> context_ptr = Context::global_instance();

  char *query_ptr = (char *)query.data_ptr();
  char *key_ptr = (char *)key.data_ptr();
  char *value_ptr = (char *)value.data_ptr();
  char *mask_ptr = (char *)mask.data_ptr();
  char *res_ptr = (char *)res.data_ptr();

  Variable *q_var = new Variable("query", g_dtype<T1>());
  q_var->set_value(query_ptr);
  Variable *k_var = new Variable("key", g_dtype<T1>());
  k_var->set_value(key_ptr);
  Variable *v_var = new Variable("value", g_dtype<T1>());
  v_var->set_value(value_ptr);
  Variable *mask_var = nullptr;  // FIXME later, only cover non mask

  SDPALayer<T1, T2> *sdpal =
      new SDPALayer<T1, T2>(max_batch_tokens, max_seq_len, head_dim, num_heads,
                            attn_prob_dropout_ratio);
  Variable *res_var = (*sdpal)(q_var, k_var, v_var, mask_var);
  res_var->set_value(res_ptr);
  sdpal->before_forward(batch_size, query_len, kv_len, kv_size, mask_future);

  context_ptr->build();
  CHECK_GPU_ERROR(cudaStreamSynchronize(0));
  CHECK_GPU_ERROR(cudaGetLastError());
  auto start = std::chrono::high_resolution_clock::now();
  sdpal->forward();
  CHECK_GPU_ERROR(cudaStreamSynchronize(0));
  CHECK_GPU_ERROR(cudaGetLastError());
  print_time_duration(start, "layer cost");
}

}  // namespace lightseq

#ifdef PYBIND_INTERFACE
#define PYBIND_MODULE_NAME TORCH_EXTENSION_NAME
#else
#define PYBIND_MODULE_NAME inference
#endif

PYBIND11_MODULE(PYBIND_MODULE_NAME, m) {
  // create default context
  lightseq::Context::create_global_context(lightseq::StatusType::Inference);

  m.def("create_global_context", &lightseq::create_global_context,
        "Create Lightseq Context");
  m.def("set_global_context", &lightseq::set_global_context,
        "Set Lightseq Context");

  m.def("create_transformer_encoder_layer_new_fp32",
        &lightseq::create_transformer_encoder_layer_new<float, float>,
        "Create LightSeq Transformer Encoder Layer with fp32");

  m.def("transformer_encoder_layer_fw_fp32",
        &lightseq::transformer_encoder_layer_fw<float, float>,
        "LightSeq Transformer Encoder forward with fp32");

  m.def("transformer_encoder_layer_bw_fp32",
        &lightseq::transformer_encoder_layer_bw<float, float>,
        "LightSeq Transformer Encoder forward with fp32");

  m.def("create_transformer_decoder_layer_new_fp32",
        &lightseq::create_transformer_decoder_layer<float, float>,
        "Create LightSeq Transformer Decoder Layer with fp32");

  m.def("transformer_decoder_layer_fw_fp32",
        &lightseq::transformer_decoder_layer_fw<float, float>,
        "LightSeq Transformer Decoder forward with fp32");

  m.def("assign_layer_weight_grad_fp32",
        &lightseq::assign_layer_weight_grad<float, float>,
        "Bind layer weights and grads");

  m.def("torch_sdpa_layer_fp32", &lightseq::torch_sdpa_layer<float, float>,
        "Empty");

  m.def("create_sdpa_layer_fp32", &lightseq::create_sdpa_layer<float, float>,
        "Create Lightseq SDPA Layer with fp32");

  m.def("sdpa_layer_fw_fp32", &lightseq::sdpa_layer_fw<float, float>,
        "Lightseq SDPA forward with fp32");

#ifdef LIGHTSEQ_cuda
  m.def("create_transformer_encoder_layer_new_fp16",
        &lightseq::create_transformer_encoder_layer_new<__half, __half>,
        "Create LightSeq Transformer Encoder Layer with fp16 (CUDA)");

  m.def("transformer_encoder_layer_fw_fp16",
        &lightseq::transformer_encoder_layer_fw<__half, __half>,
        "LightSeq Transformer Encoder forward with fp16 (CUDA)");

  m.def("transformer_encoder_layer_bw_fp16",
        &lightseq::transformer_encoder_layer_bw<__half, __half>,
        "LightSeq Transformer Encoder forward with fp16 (CUDA)");

  m.def("create_transformer_decoder_layer_new_fp16",
        &lightseq::create_transformer_decoder_layer<__half, __half>,
        "Create LightSeq Transformer Decoder Layer with fp16 (CUDA)");

  m.def("transformer_decoder_layer_fw_fp16",
        &lightseq::transformer_decoder_layer_fw<__half, __half>,
        "LightSeq Transformer Decoder forward with fp16 (CUDA)");

  m.def("assign_layer_weight_grad_fp16",
        &lightseq::assign_layer_weight_grad<__half, __half>,
        "Bind layer weights and grads");

  m.def("torch_sdpa_layer_fp16", &lightseq::torch_sdpa_layer<__half, __half>,
        "Empty");

  m.def("create_sdpa_layer_fp16", &lightseq::create_sdpa_layer<__half, __half>,
        "Create Lightseq SDPA Layer with fp16");

  m.def("sdpa_layer_fw_fp16", &lightseq::sdpa_layer_fw<__half, __half>,
        "Lightseq SDPA forward with fp16");
#endif
}
