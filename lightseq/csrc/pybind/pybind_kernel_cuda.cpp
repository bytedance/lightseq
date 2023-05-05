#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cuda.h>
#include "cuda_util.h"
#include "kernels.h"
#include "llama_kernels.h"
#include "cmath"
#include "memory"
#include <cuda.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <stdexcept>

typedef const torch::Tensor cts;
typedef torch::Tensor ts;

namespace lightseq {
namespace cuda {
template <typename T>
const T *rptr(const torch::Tensor &tensor) {
  return reinterpret_cast<const T *>(tensor.data_ptr());
}

template <typename T>
T *rptr(torch::Tensor &tensor) {
  return reinterpret_cast<T *>(tensor.data_ptr());
}

template <typename T>
void torch_launch_transform_0213(const torch::Tensor &input,
                                 torch::Tensor &output, int sz0, int sz1,
                                 int sz2, int sz3) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_transform_0213(rptr<T>(input), rptr<T>(output), sz0, sz1, sz2, sz3,
                        stream);
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
void torch_launch_quant_bias_add_transform_20314(
    torch::Tensor &output, torch::Tensor &cmask, const torch::Tensor &input,
    const torch::Tensor &bias, const torch::Tensor &cmax, int dim_0, int dim_1,
    int dim_2, int dim_3, int dim_4) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_quant_bias_add_transform_20314(
      rptr<T>(output), rptr<uint8_t>(cmask), rptr<int8_t>(input), rptr<T>(bias),
      rptr<T>(cmax), dim_0, dim_1, dim_2, dim_3, dim_4, stream);
  //   cudaStreamSynchronize(stream);
  CHECK_GPU_ERROR(cudaGetLastError());
}

template <typename T>
void torch_launch_bias_add_transform_20314_new(
    torch::Tensor &q_out, torch::Tensor &k_out, torch::Tensor &v_out,
    const torch::Tensor &input, const torch::Tensor &bias, int dim_0, int dim_1,
    int dim_2, int dim_3, int dim_4) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_bias_add_transform_20314_new(
      rptr<T>(q_out), rptr<T>(k_out), rptr<T>(v_out), rptr<T>(input),
      rptr<T>(bias), dim_0, dim_1, dim_2, dim_3, dim_4, stream);
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
void torch_launch_quant_transform4d_0213(
    torch::Tensor &output, torch::Tensor &cmask, const torch::Tensor &vals,
    const torch::Tensor &cmax, int batch_size, int seq_len, int hidden_dim,
    int nhead, int trans_count) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_quant_transform4d_0213(
      rptr<int8_t>(output), rptr<uint8_t>(cmask), rptr<T>(vals), rptr<T>(cmax),
      batch_size, seq_len, hidden_dim, nhead, trans_count, stream);
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
  //     cudaStreamSynchronize(stream);
  CHECK_GPU_ERROR(cudaGetLastError());
}

// template <typename T>
// void torch_launch_attn_softmax_new(torch::Tensor &out, torch::Tensor &inp,
//                                    const torch::Tensor &attn_mask,
//                                    int batch_size, int nhead, int from_len,
//                                    int to_len, bool is_dec_self_attn,
//                                    bool mask_future) {
//   const T *attn_mask_ptr = rptr<T>(attn_mask);
//   if (is_dec_self_attn) {
//     attn_mask_ptr = nullptr;
//   }
//   cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//   launch_attn_softmax_new(rptr<T>(out), rptr<T>(inp), attn_mask_ptr,
//   batch_size,
//                           nhead, from_len, to_len, mask_future, stream);
//   //     cudaStreamSynchronize(stream);
//   CHECK_GPU_ERROR(cudaGetLastError());
// }

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
void torch_launch_attn_softmax_bw_new(torch::Tensor &inp_grad,
                                      const torch::Tensor &out_grad,
                                      const torch::Tensor &soft_inp, int rows,
                                      int softmax_len) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_attn_softmax_bw_new(rptr<T>(inp_grad), rptr<T>(out_grad),
                             rptr<T>(soft_inp), rows, softmax_len, stream);
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
void torch_launch_ffn_bias_bwd(const torch::Tensor &inp, torch::Tensor &out,
                               int rows, int cols) {
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
void torch_launch_layer_norm_i8(torch::Tensor &ln_res, torch::Tensor &cmask,
                                torch::Tensor &vars, torch::Tensor &means,
                                const torch::Tensor &inp,
                                const torch::Tensor &scale,
                                const torch::Tensor &bias,
                                const torch::Tensor cmax, int batch_size,
                                int hidden_dim, bool with_mean) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (with_mean) {
    launch_layer_norm_i8(rptr<int8_t>(ln_res), rptr<uint8_t>(cmask),
                         rptr<T>(vars), rptr<T>(means), rptr<T>(inp),
                         rptr<T>(scale), rptr<T>(bias), rptr<T>(cmax),
                         batch_size, hidden_dim, stream);
  } else {
    launch_layer_norm_i8(rptr<int8_t>(ln_res), rptr<uint8_t>(cmask),
                         rptr<T>(vars), (T *)nullptr, rptr<T>(inp),
                         rptr<T>(scale), rptr<T>(bias), rptr<T>(cmax),
                         batch_size, hidden_dim, stream);
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

template <typename T>
void torch_launch_ln_bw_i8(
    torch::Tensor &gamma_grad, torch::Tensor &betta_grad,
    torch::Tensor &inp_grad, torch::Tensor &cmax_grad,
    const torch::Tensor &out_grad, const torch::Tensor &residual_grad,
    const torch::Tensor &inp_or_out, const torch::Tensor &gamma,
    const torch::Tensor &betta, const torch::Tensor &vars,
    const torch::Tensor &means, const torch::Tensor &cmask, int batch_size,
    int hidden_dim, bool with_mean, bool fuse_add) {
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

  launch_quant_ln_bw(rptr<T>(gamma_grad), rptr<T>(betta_grad),
                     rptr<T>(inp_grad), rptr<T>(cmax_grad), rptr<T>(out_grad),
                     p_residual_grad, rptr<T>(inp_or_out), rptr<T>(gamma),
                     p_betta, rptr<T>(vars), p_means, rptr<uint8_t>(cmask),
                     batch_size, hidden_dim, streams);
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
  cudaStreamSynchronize(stream);
  CHECK_GPU_ERROR(cudaGetLastError());
}

template <ActivationType actType, typename T>
void torch_launch_ls_dropout_act_bias(torch::Tensor &output,
                                      torch::Tensor &mask,
                                      const torch::Tensor &input,
                                      const torch::Tensor &bias, int total_seq,
                                      int hidden_dim, float ratio) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_ls_dropout_act_bias<actType, T>(
      rptr<T>(output), rptr<T>(input), rptr<uint8_t>(mask), rptr<T>(bias),
      total_seq * hidden_dim, hidden_dim, ratio, stream);
}

template <ActivationType actType, typename T>
void torch_launch_ls_dropout_act_bias_bwd(
    torch::Tensor &in_grad, torch::Tensor &bias_grad, torch::Tensor &mask,
    const torch::Tensor &input, const torch::Tensor &bias,
    const torch::Tensor &out_grad, int total_seq, int hidden_dim, float ratio) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_ls_dropout_act_bias_bwd<actType, T>(
      rptr<T>(in_grad), rptr<T>(bias_grad), rptr<T>(input), rptr<T>(bias),
      rptr<T>(out_grad), rptr<uint8_t>(mask), total_seq, hidden_dim, ratio,
      stream);
}

template <ActivationType actType, typename T>
void torch_launch_ls_quant_dropout_act_bias(
    torch::Tensor &output, torch::Tensor &cmask_out, torch::Tensor &cmask_in,
    torch::Tensor &mask, const torch::Tensor &input, const torch::Tensor &bias,
    const torch::Tensor cmax_out, const torch::Tensor cmax_in, int total_seq,
    int hidden_dim, float ratio) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_ls_quant_dropout_act_bias<actType, T>(
      rptr<int8_t>(output), rptr<uint8_t>(cmask_out), rptr<uint8_t>(cmask_in),
      rptr<uint8_t>(mask), rptr<int8_t>(input), rptr<T>(bias),
      rptr<T>(cmax_out), rptr<T>(cmax_in), total_seq * hidden_dim, hidden_dim,
      ratio, stream);
}

template <ActivationType actType, typename T>
void torch_launch_ls_quant_dropout_act_bias_bwd(
    torch::Tensor &in_grad, torch::Tensor &bias_grad,
    torch::Tensor &cmax_in_grad, torch::Tensor &cmax_out_grad,
    torch::Tensor &mask, const torch::Tensor &input,
    const torch::Tensor &cmax_in, const torch::Tensor &cmask_in,
    const torch::Tensor &cmask_out, const torch::Tensor &bias,
    const torch::Tensor &out_grad, int total_seq, int hidden_dim, float ratio) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_ls_quant_dropout_act_bias_bwd<actType, T>(
      rptr<T>(in_grad), rptr<T>(bias_grad), rptr<T>(cmax_in_grad),
      rptr<T>(cmax_out_grad), rptr<int8_t>(input), rptr<T>(cmax_in),
      rptr<uint8_t>(cmask_in), rptr<uint8_t>(cmask_out), rptr<T>(bias),
      rptr<T>(out_grad), rptr<uint8_t>(mask), total_seq, hidden_dim, ratio,
      stream);
}

template <typename T>
void torch_launch_ls_quant_bias_dropout_residual(
    torch::Tensor &output, torch::Tensor &mask, const torch::Tensor &input,
    const torch::Tensor cmax, const torch::Tensor &bias,
    const torch::Tensor &residual, int total_seq, int hidden_dim, float ratio) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_ls_quant_dropout_res_bias<T>(
      rptr<T>(output), rptr<uint8_t>(mask), rptr<int8_t>(input), rptr<T>(cmax),
      rptr<T>(bias), rptr<T>(residual), total_seq * hidden_dim, hidden_dim,
      ratio, stream);
}

template <typename T>
void torch_launch_ls_quantize(torch::Tensor &output,
                              torch::Tensor &clip_max_mask,
                              torch::Tensor &igemm_alpha,
                              const torch::Tensor &input,
                              const torch::Tensor &clip_max, int numel) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_quantize<T>(rptr<int8_t>(output), rptr<uint8_t>(clip_max_mask),
                     rptr<float>(igemm_alpha), rptr<T>(input),
                     rptr<T>(clip_max), numel, 4, stream);
}
template <typename T>
void torch_launch_ls_dequantize(torch::Tensor &output,
                                const torch::Tensor &input,
                                const torch::Tensor &clip_max, int numel) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_dequantize<T>(rptr<T>(output), rptr<int8_t>(input), rptr<T>(clip_max),
                       numel, 4, stream);
}

template <typename T>
void torch_launch_fake_quantize(torch::Tensor &clip_max_mask,
                                torch::Tensor &igemm_alpha,
                                torch::Tensor &output,
                                const torch::Tensor &input,
                                const torch::Tensor &clip_max, int numel) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_fake_quantize<T>(rptr<uint8_t>(clip_max_mask),
                          rptr<float>(igemm_alpha), rptr<T>(output),
                          rptr<T>(input), rptr<T>(clip_max), numel, 2, stream);
}

int get_sm_version() { return getSMVersion(); }

std::string get_gpu_name() { return getGPUName(); }

std::string gemm_test(int m, int n, int k) { return launch_gemm_test(m, n, k); }

template <typename T>
void torch_launch_viterbi(const torch::Tensor &start_transition,
                          const torch::Tensor &end_transition,
                          const torch::Tensor &transition,
                          const torch::Tensor &emission,
                          const torch::Tensor &mask, torch::Tensor &best_score,
                          torch::Tensor &history, torch::Tensor &best_tags,
                          int num_tags, int seq_len, int batch_size) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_viterbi(rptr<T>(start_transition), rptr<T>(end_transition),
                 rptr<T>(transition), rptr<T>(emission), rptr<T>(mask),
                 rptr<float>(best_score), rptr<int>(history),
                 rptr<int>(best_tags), num_tags, seq_len, batch_size, stream);
  cudaStreamSynchronize(stream);
  CHECK_GPU_ERROR(cudaGetLastError());
}

class RotaryPositionWeight {
 public:
  float *_device_sin_ptr;
  float *_device_cos_ptr;
  float *_sin_ptr;
  float *_cos_ptr;

  __half *_device_sin_half_ptr;
  __half *_device_cos_half_ptr;
  __half *_sin_half_ptr;
  __half *_cos_half_ptr;

  int _max_step;
  int _head_dim;

  RotaryPositionWeight(int max_step, int head_dim)
      : _max_step(max_step), _head_dim(head_dim) {
    if (head_dim & 1) {
      printf(
          "Error! head dim should be even number while using RotaryPositionQk "
          "Operator.\n");
      exit(0);
    }

    int total_size = max_step * head_dim / 2;
    _sin_ptr = (float *)malloc(total_size * sizeof(float));
    _cos_ptr = (float *)malloc(total_size * sizeof(float));

    _sin_half_ptr = (__half *)malloc(total_size * sizeof(__half));
    _cos_half_ptr = (__half *)malloc(total_size * sizeof(__half));

    for (int i = 0; i < head_dim / 2; i++) {
      float theta = std::pow(10000, -2. * i / head_dim);
      for (int j = 0; j < max_step; j++) {
        *(_sin_ptr + j * head_dim / 2 + i) =
            sin(j * theta);  // shape: [max_step, head_dim / 2]
        *(_cos_ptr + j * head_dim / 2 + i) =
            cos(j * theta);  // shape: [max_step, head_dim / 2]

        *(_sin_half_ptr + j * head_dim / 2 + i) =
            __float2half_rn(sin(j * theta));  // shape: [max_step, head_dim / 2]
        *(_cos_half_ptr + j * head_dim / 2 + i) =
            __float2half_rn(cos(j * theta));  // shape: [max_step, head_dim / 2]
      }
    }

    cudaMalloc(&_device_sin_ptr, total_size * sizeof(float));
    cudaMalloc(&_device_cos_ptr, total_size * sizeof(float));
    cudaMemcpy(_device_sin_ptr, _sin_ptr, total_size * sizeof(float),
               cudaMemcpyDefault);
    cudaMemcpy(_device_cos_ptr, _cos_ptr, total_size * sizeof(float),
               cudaMemcpyDefault);

    cudaMalloc(&_device_sin_half_ptr, total_size * sizeof(__half));
    cudaMalloc(&_device_cos_half_ptr, total_size * sizeof(__half));
    cudaMemcpy(_device_sin_half_ptr, _sin_half_ptr, total_size * sizeof(__half),
               cudaMemcpyDefault);
    cudaMemcpy(_device_cos_half_ptr, _cos_half_ptr, total_size * sizeof(__half),
               cudaMemcpyDefault);
  }
} _rotary_position_instance(2048, 128);

template <typename T>
void torch_launch_split_rotary_position(const torch::Tensor &input,
                                  torch::Tensor &q_out, torch::Tensor &cache_k_out, torch::Tensor &cache_v_out,
                                  int batch_size, int nhead, int offset_seq_len,
                                  int query_seq_len, int head_dim) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (query_seq_len + offset_seq_len > _rotary_position_instance._max_step) {
    printf(
        "Error! query_seq_len + offset_seq_len > "
        "_rotary_position_instance._max_step\n");
    return;
  }
  if (_rotary_position_instance._head_dim != head_dim) {
    printf("Error! _rotary_position_instance._head_dim != head_dim\n");
    return;
  }
  if (std::is_same<T, float>::value) {
    launch_split_rotary_position_qkv<float>(
        rptr<float>(input), _rotary_position_instance._device_sin_ptr,
        _rotary_position_instance._device_cos_ptr, rptr<float>(q_out), rptr<float>(cache_k_out), rptr<float>(cache_v_out),
        offset_seq_len + query_seq_len, batch_size, nhead, offset_seq_len, query_seq_len, head_dim,
        stream);
  } else {
    launch_split_rotary_position_qkv<__half>(
        rptr<__half>(input), _rotary_position_instance._device_sin_half_ptr,
        _rotary_position_instance._device_cos_half_ptr, rptr<__half>(q_out), rptr<__half>(cache_k_out), rptr<__half>(cache_v_out),
        offset_seq_len + query_seq_len, batch_size, nhead, offset_seq_len, query_seq_len, head_dim,
        stream);
  }
  cudaStreamSynchronize(stream);
  CHECK_GPU_ERROR(cudaGetLastError());
}

template <typename T>
void torch_silu_elewise_product(const torch::Tensor &inpA,
                                const torch::Tensor &inpB, torch::Tensor outC,
                                int batch_size, int seq_len, int inner_size) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_silu_elewise_product<T>(rptr<T>(inpA), rptr<T>(inpB), rptr<T>(outC),
                                 batch_size, seq_len, inner_size, stream);
  cudaStreamSynchronize(stream);
  CHECK_GPU_ERROR(cudaGetLastError());
}

template <typename T>
void torch_rms_layer_norm(const torch::Tensor &inp, const torch::Tensor &scale,
                          torch::Tensor &out, torch::Tensor &rms_out,
                          int batch_tokens, int hidden_dim,
                          const float epsilon = 1e-6) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_rms_layer_norm<T>(rptr<T>(inp), rptr<T>(scale), rptr<T>(out),
                           rptr<T>(rms_out), batch_tokens, hidden_dim, stream,
                           epsilon);
  cudaStreamSynchronize(stream);
  CHECK_GPU_ERROR(cudaGetLastError());
}

}  // namespace cuda
}  // namespace lightseq

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  //   lightseq::cuda::_rotary_position_instance =
  //   lightseq::cuda::RotaryPositionWeight(2048, 128);

  m.def("torch_launch_transform_0213_fp32",
        &lightseq::cuda::torch_launch_transform_0213<float>,
        "Test kernel wrapper");
  m.def("torch_launch_transform_0213_fp16",
        &lightseq::cuda::torch_launch_transform_0213<__half>,
        "Test kernel wrapper");
  m.def("torch_launch_bias_add_transform_20314_fp32",
        &lightseq::cuda::torch_launch_bias_add_transform_20314<float>,
        "Test kernel wrapper");
  m.def("torch_launch_bias_add_transform_20314_fp16",
        &lightseq::cuda::torch_launch_bias_add_transform_20314<__half>,
        "Test kernel wrapper");

  m.def("torch_launch_transform4d_0213_fp32",
        &lightseq::cuda::torch_launch_transform4d_0213<float>,
        "Test kernel wrapper");
  m.def("torch_launch_transform4d_0213_fp16",
        &lightseq::cuda::torch_launch_transform4d_0213<__half>,
        "Test kernel wrapper");

  m.def("torch_launch_bias_add_transform_20314_new_fp32",
        &lightseq::cuda::torch_launch_bias_add_transform_20314_new<float>,
        "Test kernel wrapper");
  m.def("torch_launch_bias_add_transform_20314_new_fp16",
        &lightseq::cuda::torch_launch_bias_add_transform_20314_new<__half>,
        "Test kernel wrapper");

  m.def("torch_launch_fused_add2_fp32",
        &lightseq::cuda::torch_launch_fused_add2<float>, "Test kernel wrapper");
  m.def("torch_launch_fused_add2_fp16",
        &lightseq::cuda::torch_launch_fused_add2<__half>,
        "Test kernel wrapper");
  m.def("torch_launch_ffn_bias_bwd_fp32",
        &lightseq::cuda::torch_launch_ffn_bias_bwd<float>,
        "Test kernel wrapper");
  m.def("torch_launch_ffn_bias_bwd_fp16",
        &lightseq::cuda::torch_launch_ffn_bias_bwd<__half>,
        "Test kernel wrapper");
  m.def("torch_launch_attn_softmax_fp32",
        &lightseq::cuda::torch_launch_attn_softmax<float>,
        "Test kernel wrapper");
  m.def("torch_launch_attn_softmax_fp16",
        &lightseq::cuda::torch_launch_attn_softmax<__half>,
        "Test kernel wrapper");
  //   m.def("torch_launch_attn_softmax_new_fp32",
  //         &lightseq::cuda::torch_launch_attn_softmax_new<float>,
  //         "Test kernel wrapper");
  //   m.def("torch_launch_attn_softmax_new_fp16",
  //         &lightseq::cuda::torch_launch_attn_softmax_new<__half>,
  //         "Test kernel wrapper");
  m.def("torch_launch_attn_softmax_bw_fp32",
        &lightseq::cuda::torch_launch_attn_softmax_bw<float>,
        "Test kernel wrapper");
  m.def("torch_launch_attn_softmax_bw_fp16",
        &lightseq::cuda::torch_launch_attn_softmax_bw<__half>,
        "Test kernel wrapper");

  m.def("torch_launch_attn_softmax_bw_new_fp32",
        &lightseq::cuda::torch_launch_attn_softmax_bw_new<float>,
        "Test kernel wrapper");
  m.def("torch_launch_attn_softmax_bw_new_fp16",
        &lightseq::cuda::torch_launch_attn_softmax_bw_new<__half>,
        "Test kernel wrapper");

  m.def("torch_launch_layer_norm_fp32",
        &lightseq::cuda::torch_launch_layer_norm<float>, "Test kernel wrapper");
  m.def("torch_launch_layer_norm_fp16",
        &lightseq::cuda::torch_launch_layer_norm<__half>,
        "Test kernel wrapper");
  m.def("torch_launch_layer_norm_i8_fp32",
        &lightseq::cuda::torch_launch_layer_norm_i8<float>,
        "Test kernel wrapper");
  m.def("torch_launch_layer_norm_i8_fp16",
        &lightseq::cuda::torch_launch_layer_norm_i8<__half>,
        "Test kernel wrapper");
  m.def("torch_launch_ln_bw_fp32", &lightseq::cuda::torch_launch_ln_bw<float>,
        "Test kernel wrapper");
  m.def("torch_launch_ln_bw_fp16", &lightseq::cuda::torch_launch_ln_bw<__half>,
        "Test kernel wrapper");
  m.def("torch_launch_ln_bw_i8_fp32",
        &lightseq::cuda::torch_launch_ln_bw_i8<float>, "Test kernel wrapper");
  m.def("torch_launch_ln_bw_i8_fp16",
        &lightseq::cuda::torch_launch_ln_bw_i8<__half>, "Test kernel wrapper");
  m.def("torch_launch_curand_init", &lightseq::cuda::torch_curand_init,
        "Test kernel wrapper");
  m.def("torch_launch_concat3_dim1_fp32",
        &lightseq::cuda::torch_launch_concat3_dim1<float>,
        "Test kernel wrapper");
  m.def("torch_launch_concat3_dim1_fp16",
        &lightseq::cuda::torch_launch_concat3_dim1<__half>,
        "Test kernel wrapper");
  m.def("torch_launch_ls_dropout_relu_bias_fp32",
        &lightseq::cuda::torch_launch_ls_dropout_act_bias<
            lightseq::ActivationType::kRelu, float>,
        "Test kernel wrapper");
  m.def("torch_launch_ls_dropout_relu_bias_fp16",
        &lightseq::cuda::torch_launch_ls_dropout_act_bias<
            lightseq::ActivationType::kRelu, __half>,
        "Test kernel wrapper");
  m.def("torch_launch_ls_dropout_gelu_bias_fp32",
        &lightseq::cuda::torch_launch_ls_dropout_act_bias<
            lightseq::ActivationType::kGelu, float>,
        "Test kernel wrapper");
  m.def("torch_launch_ls_dropout_gelu_bias_fp16",
        &lightseq::cuda::torch_launch_ls_dropout_act_bias<
            lightseq::ActivationType::kGelu, __half>,
        "Test kernel wrapper");
  m.def("torch_launch_ls_dropout_relu_bias_bwd_fp32",
        &lightseq::cuda::torch_launch_ls_dropout_act_bias_bwd<
            lightseq::ActivationType::kRelu, float>,
        "Test kernel wrapper");
  m.def("torch_launch_ls_dropout_relu_bias_bwd_fp16",
        &lightseq::cuda::torch_launch_ls_dropout_act_bias_bwd<
            lightseq::ActivationType::kRelu, __half>,
        "Test kernel wrapper");
  m.def("torch_launch_ls_dropout_gelu_bias_bwd_fp32",
        &lightseq::cuda::torch_launch_ls_dropout_act_bias_bwd<
            lightseq::ActivationType::kGelu, float>,
        "Test kernel wrapper");
  m.def("torch_launch_ls_dropout_gelu_bias_bwd_fp16",
        &lightseq::cuda::torch_launch_ls_dropout_act_bias_bwd<
            lightseq::ActivationType::kGelu, __half>,
        "Test kernel wrapper");
  m.def("torch_launch_ls_quant_dropout_relu_bias_fp32",
        &lightseq::cuda::torch_launch_ls_quant_dropout_act_bias<
            lightseq::ActivationType::kRelu, float>,
        "Test kernel wrapper");
  m.def("torch_launch_ls_quant_dropout_relu_bias_fp16",
        &lightseq::cuda::torch_launch_ls_quant_dropout_act_bias<
            lightseq::ActivationType::kRelu, __half>,
        "Test kernel wrapper");
  m.def("torch_launch_ls_quant_dropout_gelu_bias_fp32",
        &lightseq::cuda::torch_launch_ls_quant_dropout_act_bias<
            lightseq::ActivationType::kGelu, float>,
        "Test kernel wrapper");
  m.def("torch_launch_ls_quant_dropout_gelu_bias_fp16",
        &lightseq::cuda::torch_launch_ls_quant_dropout_act_bias<
            lightseq::ActivationType::kGelu, __half>,
        "Test kernel wrapper");
  m.def("torch_launch_ls_quant_dropout_relu_bias_bwd_fp32",
        &lightseq::cuda::torch_launch_ls_quant_dropout_act_bias_bwd<
            lightseq::ActivationType::kRelu, float>,
        "Test kernel wrapper");
  m.def("torch_launch_ls_quant_dropout_relu_bias_bwd_fp16",
        &lightseq::cuda::torch_launch_ls_quant_dropout_act_bias_bwd<
            lightseq::ActivationType::kRelu, __half>,
        "Test kernel wrapper");
  m.def("torch_launch_ls_quant_dropout_gelu_bias_bwd_fp32",
        &lightseq::cuda::torch_launch_ls_quant_dropout_act_bias_bwd<
            lightseq::ActivationType::kGelu, float>,
        "Test kernel wrapper");
  m.def("torch_launch_ls_quant_dropout_gelu_bias_bwd_fp16",
        &lightseq::cuda::torch_launch_ls_quant_dropout_act_bias_bwd<
            lightseq::ActivationType::kGelu, __half>,
        "Test kernel wrapper");
  m.def("torch_launch_ls_quant_bias_dropout_residual_fp32",
        &lightseq::cuda::torch_launch_ls_quant_bias_dropout_residual<float>,
        "Test kernel wrapper");
  m.def("torch_launch_ls_quant_bias_dropout_residual_fp16",
        &lightseq::cuda::torch_launch_ls_quant_bias_dropout_residual<__half>,
        "Test kernel wrapper");
  m.def("torch_launch_quant_bias_add_transform_20314_fp32",
        &lightseq::cuda::torch_launch_quant_bias_add_transform_20314<float>,
        "Test kernel wrapper");
  m.def("torch_launch_quant_bias_add_transform_20314_fp16",
        &lightseq::cuda::torch_launch_quant_bias_add_transform_20314<__half>,
        "Test kernel wrapper");
  m.def("torch_launch_quant_transform4d_0213_fp32",
        &lightseq::cuda::torch_launch_quant_transform4d_0213<float>,
        "Test kernel wrapper");
  m.def("torch_launch_quant_transform4d_0213_fp16",
        &lightseq::cuda::torch_launch_quant_transform4d_0213<__half>,
        "Test kernel wrapper");
  m.def("torch_launch_ls_quantize_fp32",
        &lightseq::cuda::torch_launch_ls_quantize<float>,
        "Test kernel wrapper");
  m.def("torch_launch_ls_quantize_fp16",
        &lightseq::cuda::torch_launch_ls_quantize<__half>,
        "Test kernel wrapper");
  m.def("torch_launch_ls_dequantize_fp32",
        &lightseq::cuda::torch_launch_ls_dequantize<float>,
        "Test kernel wrapper");
  m.def("torch_launch_ls_dequantize_fp16",
        &lightseq::cuda::torch_launch_ls_dequantize<__half>,
        "Test kernel wrapper");
  m.def("torch_launch_fake_quantize_fp32",
        &lightseq::cuda::torch_launch_fake_quantize<float>,
        "Test kernel wrapper");
  m.def("torch_launch_fake_quantize_fp16",
        &lightseq::cuda::torch_launch_fake_quantize<__half>,
        "Test kernel wrapper");
  m.def("get_sm_version", &lightseq::cuda::get_sm_version,
        "Test kernel wrapper");
  m.def("get_gpu_name", &lightseq::cuda::get_gpu_name, "Test kernel wrapper");
  m.def("gemm_test", &lightseq::cuda::gemm_test, "Test kernel wrapper");
  m.def("torch_launch_viterbi_fp16",
        &lightseq::cuda::torch_launch_viterbi<__half>, "Test kernel wrapper");
  m.def("torch_launch_viterbi_fp32",
        &lightseq::cuda::torch_launch_viterbi<float>, "Test kernel wrapper");

  m.def("torch_launch_split_rotary_position_fp32",
        &lightseq::cuda::torch_launch_split_rotary_position<float>,
        "Test llama rotary position kernel");
  m.def("torch_launch_split_rotary_position_fp16",
        &lightseq::cuda::torch_launch_split_rotary_position<half>,
        "Test llama rotary position kernel");
  m.def("torch_silu_elewise_product_fp32",
        &lightseq::cuda::torch_silu_elewise_product<float>,
        "Test llama rotary position kernel");
  m.def("torch_silu_elewise_product_fp16",
        &lightseq::cuda::torch_silu_elewise_product<__half>,
        "Test llama rotary position kernel");

  m.def("torch_rms_layer_norm_fp32",
        &lightseq::cuda::torch_rms_layer_norm<float>,
        "Test llama rms layer norm kernel");
  m.def("torch_rms_layer_norm_fp16",
        &lightseq::cuda::torch_rms_layer_norm<__half>,
        "Test llama rms layer norm kernel");
}
