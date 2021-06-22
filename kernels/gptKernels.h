#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>

#include "3rdparty/cub/cub/cub.cuh"

namespace lightseq {
namespace cuda {

template <typename T>
void ker_gpt_embedding_launcher(int batch_size, int batch_seq_len,
                                int hidden_size, cudaStream_t stream,
                                const T* token_emb, const T* pos_emb,
                                const int* token_id, T* output,
                                int* real_seq_len, int padding_id,
                                int pos_offset);

template <typename T>
void ker_correlation_softmax_gpt_launcher(int batch_size, int batch_seq_len,
                                          int head_num, cudaStream_t stream,
                                          T* correlation,
                                          const int* real_seq_len);

template <typename T>
void ker_attention_mask_weights_launcher(int batch_size, int dst_seq_len,
                                         int src_seq_len, int head_num,
                                         cudaStream_t stream, T* correlation,
                                         const int* real_seq_len);

template <typename T>
void ker_arrange_qkv_with_cache_launcher(int batch_token_num, int hidden_size,
                                         cudaStream_t stream, const T* ori_qkv,
                                         const T* qkv_bias, T* new_q, T* new_k,
                                         T* k_cache, T* new_v, T* v_cache,
                                         int max_batch_dim, int batch_seq_len,
                                         int dim_per_head, int head_num);

template <typename T>
void ker_ppl_launcher(int batch_size, int batch_seq_len,
                      int max_thread_per_block, cudaStream_t stream,
                      const T* logits, const int* input_ids,
                      const int* real_seq_len, float* ppl, int vocab_size);

template <typename T>
void ker_topk_sample_launcher(int batch_size, int batch_seq_len,
                              int logits_seq_len, int max_thread_per_block,
                              cudaStream_t stream, const T* logits,
                              int* old_input_ids, int* new_input_ids,
                              const int* real_seq_len, const int vocab_size,
                              const int k, int* all_finished,
                              curandState* curandstate, int eos_id);

template <typename T>
void ker_topp_sample_launcher(int batch_size, int batch_seq_len,
                              int logits_seq_len, int max_thread_per_block,
                              cudaStream_t stream, const T* logits,
                              int* old_input_ids, int* new_input_ids,
                              const int* real_seq_len, const int vocab_size,
                              const float p, int* unfinished,
                              curandState* curandstate, int eos_id);

}  // namespace cuda
}  // namespace lightseq
