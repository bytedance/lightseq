#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <string>

#include "declaration.h"
#include "context.h"
#include "normalize_layer.h"
#include "feed_forward.h"

// x is torch::Tensor
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

namespace lightseq {

template <typename T1, typename T2>
void layer_normalize_fw(const torch::Tensor& ln_res, const torch::Tensor& inp,
                        const torch::Tensor& gamma, const torch::Tensor& betta,
                        int hidden_dim, int batch_tokens) {
  Context::new_thread_context();

  T1* ln_res_ptr = (T1*)ln_res.data_ptr();
  T1* inp_ptr = (T1*)inp.data_ptr();
  T1* gamma_ptr = (T1*)gamma.data_ptr();
  T1* betta_ptr = (T1*)betta.data_ptr();

  Variable* inp_var = new Variable("inp", (char*)inp_ptr);
  Variable* gamma_var = new Variable("gamma", (char*)gamma_ptr);
  Variable* betta_var = new Variable("betta", (char*)betta_ptr);

  NormalizeLayerOp<T1, T2>* op =
      new NormalizeLayerOp<T1, T2>(batch_tokens, hidden_dim);

  Variable* out = (*op)(inp_var, gamma_var, betta_var);

  out->set_value((char*)ln_res_ptr);

  op->before_forward(batch_tokens);

  thread_context_ptr->build();

  op->forward();

  // Context::remove_thread_context();
}

template <typename T1, typename T2>
void layer_normalize_bw(const torch::Tensor& ln_res,
                        const torch::Tensor& ln_res_grad,
                        const torch::Tensor& inp, const torch::Tensor& inp_grad,
                        const torch::Tensor& gamma,
                        const torch::Tensor& gamma_grad,
                        const torch::Tensor& betta,
                        const torch::Tensor& betta_grad, int hidden_dim,
                        int batch_tokens) {
  Context::new_thread_context();

  T1* ln_res_ptr = (T1*)ln_res.data_ptr();
  T2* ln_res_grad_ptr = (T2*)ln_res_grad.data_ptr();

  T1* inp_ptr = (T1*)inp.data_ptr();
  T2* inp_grad_ptr = (T2*)inp_grad.data_ptr();

  T1* gamma_ptr = (T1*)gamma.data_ptr();
  T2* gamma_grad_ptr = (T2*)gamma_grad.data_ptr();

  T1* betta_ptr = (T1*)betta.data_ptr();
  T2* betta_grad_ptr = (T2*)betta_grad.data_ptr();

  Variable* inp_var = new Variable("inp", (char*)inp_ptr, (char*)inp_grad_ptr);
  Variable* gamma_var =
      new Variable("gamma", (char*)gamma_ptr, (char*)gamma_grad_ptr);
  Variable* betta_var =
      new Variable("betta", (char*)betta_ptr, (char*)betta_grad_ptr);

  NormalizeLayerOp<T1, T2>* op =
      new NormalizeLayerOp<T1, T2>(batch_tokens, hidden_dim);

  Variable* out = (*op)(inp_var, gamma_var, betta_var);

  out->set_value((char*)ln_res_ptr);
  out->set_grad((char*)ln_res_grad_ptr);

  op->before_forward(batch_tokens);

  thread_context_ptr->build();

  op->forward();

  op->backward();

  // Context::remove_thread_context();
}

template <typename T1, typename T2>
void feed_forward_fw(const torch::Tensor& inp, const torch::Tensor& weights,
                     const torch::Tensor& out, int output_dim, int input_dim,
                     int batch_tokens) {
  Context::new_thread_context();

  T1* inp_ptr = (T1*)inp.data_ptr();
  T1* weights_ptr = (T1*)weights.data_ptr();
  T1* out_ptr = (T1*)out.data_ptr();

  Variable* inp_var = new Variable("inp", (char*)inp_ptr);
  Variable* weight_var = new Variable("weights", (char*)weights_ptr);

  FeedForwardOp<T1, T2>* op =
      new FeedForwardOp<T1, T2>(batch_tokens, output_dim, input_dim);

  Variable* op_out = (*op)(inp_var, weight_var);

  op_out->set_value((char*)out_ptr);

  op->before_forward(batch_tokens);

  // printf("Running Step.0\n");
  thread_context_ptr->build();

  op->forward();

  // Context::remove_thread_context();
}

}  // namespace lightseq

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("layer_normalize_fw_fp32", &lightseq::layer_normalize_fw<float, float>,
        "Calculate LightSeq Layer Normalize  with fp32 (CUDA)");
  m.def("layer_normalize_fw_fp16",
        &lightseq::layer_normalize_fw<__half, __half>,
        "Calculate LightSeq Layer Normalize  with fp16 (CUDA)");

  m.def("layer_normalize_bw_fp32", &lightseq::layer_normalize_bw<float, float>,
        "Calculate LightSeq Layer Normalize  with fp32 (CUDA)");
  m.def("layer_normalize_bw_fp16",
        &lightseq::layer_normalize_bw<__half, __half>,
        "Calculate LightSeq Layer Normalize  with fp16 (CUDA)");

  m.def("feed_forward_fw_fp32", &lightseq::feed_forward_fw<float, float>,
        "Calculate LightSeq FeedForward with fp32 (CUDA)");
  m.def("feed_forward_fw_fp16", &lightseq::feed_forward_fw<__half, __half>,
        "Calculate LightSeq FeedForward with fp16 (CUDA)");
}
