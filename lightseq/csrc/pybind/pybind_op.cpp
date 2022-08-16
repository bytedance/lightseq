#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <string>

#include "declaration.h"
#include "context.h"
#include "normalize_layer.h"

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

  size_t batch_dim = batch_tokens * hidden_dim;

  Variable* inp_var = new Variable("inp", (char*)inp_ptr);
  Variable* gamma_var = new Variable("gamma", (char*)gamma_ptr);
  Variable* betta_var = new Variable("betta", (char*)betta_ptr);

  NormalizeLayerOp<T1, T2>* op =
      new NormalizeLayerOp<T1, T2>(batch_tokens, hidden_dim);

  Variable* out = (*op)(inp_var, gamma_var, betta_var);

  out->set_value((char*)ln_res_ptr);

  op->before_forward(batch_tokens);

  op->forward();

  Context::remove_thread_context();
}

}  // namespace lightseq

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("layer_normalize_fw_fp32", &lightseq::layer_normalize_fw<float, float>,
        "Create LightSeq Cross Entropy Layer with fp32 (CUDA)");
  m.def("layer_normalize_fw_fp16",
        &lightseq::layer_normalize_fw<__half, __half>,
        "Create LightSeq Cross Entropy Layer with fp16 (CUDA)");
}
