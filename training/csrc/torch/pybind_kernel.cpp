#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cuda.h>

#include "cuda_util.h"
#include "kernels.h"

typedef const torch::Tensor cts;
typedef torch::Tensor ts;

template <typename T>
const T *rptr(const torch::Tensor &tensor) {
  return reinterpret_cast<const T *>(tensor.data_ptr());
}

template <typename T>
T *rptr(torch::Tensor &tensor) {
  return reinterpret_cast<T *>(tensor.data_ptr());
}

template <typename T>
void torch_launch_transform_0213(torch::Tensor &output,
                                 const torch::Tensor &vals, int batch_size,
                                 int seq_len, int hidden_dim, int nhead) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_transform_0213(rptr<T>(output), rptr<T>(vals), batch_size, seq_len,
                        hidden_dim, nhead, stream);
  //   cudaStreamSynchronize(stream);
  CHECK_GPU_ERROR(cudaGetLastError());
}

template <typename T>
void torch_launch_bias_add_transform_20314(torch::Tensor &output,
                                           const torch::Tensor &input,
                                           const torch::Tensor &bias, int dim_0,
                                           int dim_1, int dim_2, int dim_3,
                                           int dim_4) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_bias_add_transform_20314(rptr<T>(output), rptr<T>(input),
                                  rptr<T>(bias), dim_0, dim_1, dim_2, dim_3,
                                  dim_4, stream);
  //   cudaStreamSynchronize(stream);
  CHECK_GPU_ERROR(cudaGetLastError());
}

template <typename T>
void torch_launch_transform4d_0213(torch::Tensor &output,
                                   const torch::Tensor &vals, int batch_size,
                                   int seq_len, int hidden_dim, int nhead,
                                   int trans_count) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_transform4d_0213(rptr<T>(output), rptr<T>(vals), batch_size, seq_len,
                          hidden_dim, nhead, trans_count, stream);
  //   cudaStreamSynchronize(stream);
  CHECK_GPU_ERROR(cudaGetLastError());
}

template <typename T>
void torch_launch_attn_softmax(torch::Tensor &vals,
                               const torch::Tensor &attn_mask, int batch_size,
                               int nhead, int from_len, int to_len,
                               bool is_dec_self_attn, bool mask_future) {
  const T *attn_mask_ptr = rptr<T>(attn_mask);
  if (is_dec_self_attn) {
    attn_mask_ptr = nullptr;
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_attn_softmax(rptr<T>(vals), attn_mask_ptr, batch_size, nhead, from_len,
                      to_len, mask_future, stream);
  //   cudaStreamSynchronize(stream);
  CHECK_GPU_ERROR(cudaGetLastError());
}

template <typename T>
void torch_launch_attn_softmax_bw(torch::Tensor &out_grad,
                                  const torch::Tensor &soft_inp, int rows,
                                  int softmax_len) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_attn_softmax_bw(rptr<T>(out_grad), rptr<T>(soft_inp), rows,
                         softmax_len, stream);
  //   cudaStreamSynchronize(stream);
  CHECK_GPU_ERROR(cudaGetLastError());
}

template <typename T>
void torch_launch_fused_add2(torch::Tensor &out, const torch::Tensor &inp1,
                             const torch::Tensor &inp2, int batch_size,
                             int seq_len, int hidden_dim) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_fused_add2(rptr<T>(out), rptr<T>(inp1), rptr<T>(inp2), batch_size,
                    seq_len, hidden_dim, stream);
  //     cudaStreamSynchronize(stream);
  CHECK_GPU_ERROR(cudaGetLastError());
}

template <typename T>
void torch_launch_fuse_transpose_bias_kernel(const torch::Tensor &inp,
                                             torch::Tensor &out, int rows,
                                             int cols) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_fuse_transpose_bias_kernel(rptr<T>(inp), rptr<T>(out), rows, cols,
                                    stream);
  //     cudaStreamSynchronize(stream);
  CHECK_GPU_ERROR(cudaGetLastError());
}

template <typename T>
void torch_launch_layer_norm(torch::Tensor &ln_res, torch::Tensor &vars,
                             torch::Tensor &means, const torch::Tensor &inp,
                             const torch::Tensor &scale,
                             const torch::Tensor &bias, int batch_size,
                             int hidden_dim, bool with_mean) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (with_mean) {
    launch_layer_norm(rptr<T>(ln_res), rptr<T>(vars), rptr<T>(means),
                      rptr<T>(inp), rptr<T>(scale), rptr<T>(bias), batch_size,
                      hidden_dim, stream);
  } else {
    launch_layer_norm(rptr<T>(ln_res), rptr<T>(vars), (T *)nullptr,
                      rptr<T>(inp), rptr<T>(scale), rptr<T>(bias), batch_size,
                      hidden_dim, stream);
  }
}

template <typename T>
void torch_launch_ln_bw(torch::Tensor &gamma_grad, torch::Tensor &betta_grad,
                        torch::Tensor &inp_grad, const torch::Tensor &out_grad,
                        const torch::Tensor &residual_grad,
                        const torch::Tensor &inp_or_out,
                        const torch::Tensor &gamma, const torch::Tensor &betta,
                        const torch::Tensor &vars, const torch::Tensor &means,
                        int batch_size, int hidden_dim, bool with_mean,
                        bool fuse_add) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cudaStream_t streams[2] = {stream, stream};
  const T *p_residual_grad;
  const T *p_betta;
  const T *p_means;

  if (fuse_add) {
    p_residual_grad = rptr<T>(residual_grad);
  } else {
    p_residual_grad = nullptr;
  }
  if (with_mean) {
    p_means = rptr<T>(means);
    p_betta = nullptr;
  } else {
    p_means = nullptr;
    p_betta = rptr<T>(betta);
  }

  launch_ln_bw(rptr<T>(gamma_grad), rptr<T>(betta_grad), rptr<T>(inp_grad),
               rptr<T>(out_grad), p_residual_grad, rptr<T>(inp_or_out),
               rptr<T>(gamma), p_betta, rptr<T>(vars), p_means, batch_size,
               hidden_dim, streams);
}

void torch_curand_init(int batch_size, int hidden_dim) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_curand_init(batch_size * hidden_dim, hidden_dim, stream);
}

template <typename T>
void torch_launch_concat3_dim1(const torch::Tensor &inp1,
                               const torch::Tensor &inp2, torch::Tensor &output,
                               int sz0, int sz2, int sz1_1, int sz1_2) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_concat3_dim1(rptr<T>(inp1), rptr<T>(inp2), rptr<T>(output), sz0, sz2,
                      sz1_1, sz1_2, stream);
  // cudaStreamSynchronize(stream);
  CHECK_GPU_ERROR(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("torch_launch_transform_0213_fp32", &torch_launch_transform_0213<float>,
        "Test kernel wrapper");
  m.def("torch_launch_transform_0213_fp16",
        &torch_launch_transform_0213<__half>, "Test kernel wrapper");
  m.def("torch_launch_bias_add_transform_20314_fp32",
        &torch_launch_bias_add_transform_20314<float>, "Test kernel wrapper");
  m.def("torch_launch_bias_add_transform_20314_fp16",
        &torch_launch_bias_add_transform_20314<__half>, "Test kernel wrapper");
  m.def("torch_launch_transform4d_0213_fp32",
        &torch_launch_transform4d_0213<float>, "Test kernel wrapper");
  m.def("torch_launch_transform4d_0213_fp16",
        &torch_launch_transform4d_0213<__half>, "Test kernel wrapper");
  m.def("torch_launch_fused_add2_fp32", &torch_launch_fused_add2<float>,
        "Test kernel wrapper");
  m.def("torch_launch_fused_add2_fp16", &torch_launch_fused_add2<__half>,
        "Test kernel wrapper");
  m.def("torch_launch_fuse_transpose_bias_kernel_fp32",
        &torch_launch_fuse_transpose_bias_kernel<float>, "Test kernel wrapper");
  m.def("torch_launch_fuse_transpose_bias_kernel_fp16",
        &torch_launch_fuse_transpose_bias_kernel<__half>,
        "Test kernel wrapper");
  m.def("torch_launch_attn_softmax_fp32", &torch_launch_attn_softmax<float>,
        "Test kernel wrapper");
  m.def("torch_launch_attn_softmax_fp16", &torch_launch_attn_softmax<__half>,
        "Test kernel wrapper");
  m.def("torch_launch_attn_softmax_bw_fp32",
        &torch_launch_attn_softmax_bw<float>, "Test kernel wrapper");
  m.def("torch_launch_attn_softmax_bw_fp16",
        &torch_launch_attn_softmax_bw<__half>, "Test kernel wrapper");
  m.def("torch_launch_layer_norm_fp32", &torch_launch_layer_norm<float>,
        "Test kernel wrapper");
  m.def("torch_launch_layer_norm_fp16", &torch_launch_layer_norm<__half>,
        "Test kernel wrapper");
  m.def("torch_launch_ln_bw_fp32", &torch_launch_ln_bw<float>,
        "Test kernel wrapper");
  m.def("torch_launch_ln_bw_fp16", &torch_launch_ln_bw<__half>,
        "Test kernel wrapper");
  m.def("torch_launch_curand_init", &torch_curand_init, "Test kernel wrapper");
  m.def("torch_launch_concat3_dim1_fp32", &torch_launch_concat3_dim1<float>,
        "Test kernel wrapper");
  m.def("torch_launch_concat3_dim1_fp16", &torch_launch_concat3_dim1<__half>,
        "Test kernel wrapper");
}
