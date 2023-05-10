#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include "kernels.h"
#include <stdexcept>

namespace lightseq {
namespace cuda {

template <typename T>
void launch_llama_embedding(const T *token_emb, const int *tokens, T *output,
                            T *pad_mask_ptr, int *left_pad_len_ptr,
                            int batch_size, int beam_size, int hidden_dim,
                            int step_offset, int seq_len, int max_step,
                            int padding_id, cudaStream_t stream);

template <typename T>
void launch_split_rotary_position_qkv(const T *input_ptr, const T *sin_ptr,
                                      const T *cos_ptr, T *q_out,
                                      T *cache_k_out, T *cache_v_out,
                                      size_t max_step, size_t batch_size,
                                      size_t nhead, size_t offset_seq_len,
                                      size_t query_len, size_t head_dim,
                                      cudaStream_t stream);

template <typename T>
void launch_silu_elewise_product(const T *inp_ptr, T *out_ptr,
                                 size_t batch_size, size_t seq_len,
                                 size_t inner_size, cudaStream_t stream);

template <typename T>
void launch_rms_layer_norm(const T *inp_ptr, const T *scale_ptr, T *out_ptr,
                           T *res_ptr, T *rms_ptr, size_t batch_tokens,
                           size_t hidden_dim, cudaStream_t stream,
                           const float ln_epsilon = 1e-6f);

}  // namespace cuda
}  // namespace lightseq
