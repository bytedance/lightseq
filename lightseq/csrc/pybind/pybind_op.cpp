#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <string>

#include "declaration.h"
#include "context.h"

#include "split_head_op.h"

// x is torch::Tensor
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

namespace lightseq {

template <typename T1, typename T2>
void torch_split_head_op(const torch::Tensor& input, const torch::Tensor& bias,
                         torch::Tensor& query, torch::Tensor& key,
                         torch::Tensor& value, int batch_size, int hidden_dim,
                         int num_heads, int seq_len, int qkv_num, int cache_sz,
                         int step) {
  Context::create_global_context(StatusType::Inference);
  std::shared_ptr<Context> context_ptr = Context::global_instance();

  char* input_ptr = (char*)input.data_ptr();
  char* bias_ptr = (char*)bias.data_ptr();
  char* query_ptr = (char*)query.data_ptr();
  char* key_ptr = (char*)key.data_ptr();
  char* value_ptr = (char*)value.data_ptr();

  Variable* input_var = new Variable("input", g_dtype<T1>());
  input_var->set_value(input_ptr);
  Variable* bias_var = new Variable("bias", g_dtype<T1>());
  bias_var->set_value(bias_ptr);
  SplitHeadOp<T1, T2>* op = new SplitHeadOp<T1, T2>(
      batch_size * seq_len, num_heads, hidden_dim, qkv_num, cache_sz);

  if (cache_sz > 0) {
    // with cache
    Variable* vk = new Variable("key", g_dtype<T1>());
    vk->set_value(key_ptr);
    Variable* vv = new Variable("value", g_dtype<T1>());
    vv->set_value(value_ptr);
    Variable* vq = (*op)(input_var, bias_var, vk, vv);
    vq->set_value(query_ptr);
    op->before_forward(batch_size, seq_len, step);
  } else {
    // without cache
    std::vector<Variable*> res = (*op)(input_var, bias_var);
    res[0]->set_value(query_ptr);
    if (qkv_num == 3) {
      res[1]->set_value(key_ptr);
      res[2]->set_value(value_ptr);
    }
    op->before_forward(batch_size, seq_len);
  }

  context_ptr->build();
  CHECK_GPU_ERROR(cudaGetLastError());
  auto start = std::chrono::high_resolution_clock::now();
  op->forward();
  CHECK_GPU_ERROR(cudaGetLastError());
  print_time_duration(start, "op cost");
}

}  // namespace lightseq

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("torch_split_head_op_fp32",
        &lightseq::torch_split_head_op<float, float>, "empty");
  m.def("torch_split_head_op_fp16",
        &lightseq::torch_split_head_op<__half, __half>, "empty");
}
