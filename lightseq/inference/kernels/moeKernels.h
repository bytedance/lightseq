#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>

namespace lightseq {
namespace cuda {

template <typename T>
void ker_norm_layer_prepost_launcher(int token_num, int hidden_size,
                                     cudaStream_t stream, T* input, T* output,
                                     const T* scale, const T* bias,
                                     const int max_thread_per_block,
                                     bool is_post_ln = false);

template <typename T>
void ker_softmax_topk_router_launcher(int batch_token_num, int expert_num,
                                      int max_token_num, int topk,
                                      cudaStream_t stream, const T* gate_out,
                                      float* score_routed, int* expert_routed);

template <typename T>
void ker_reorder_tokens_launcher(int batch_token_num, int expert_num,
                                 int max_token_num, int hidden_size,
                                 int max_thread_per_block, cudaStream_t stream,
                                 const T* input, const float* score, T* output);

template <typename T>
void ker_strided_bias_gelu_launcher(int batch_token_num, int expert_num,
                                    int max_token_num, int feature_dim,
                                    int block_dim, cudaStream_t stream,
                                    T* input, const T* bias);

template <typename T>
void ker_strided_bias_relu_launcher(int batch_token_num, int expert_num,
                                    int max_token_num, int feature_dim,
                                    int block_dim, cudaStream_t stream,
                                    T* input, const T* bias);

template <typename T>
void ker_bias_redirect_residual_launcher(int hidden_size, int max_token_num,
                                         int topk, int batch_token_num,
                                         int block_dim, cudaStream_t stream,
                                         const T* input, const T* bias,
                                         const float* score,
                                         const int* expert_routed, T* output);

}  // namespace cuda
}  // namespace lightseq
