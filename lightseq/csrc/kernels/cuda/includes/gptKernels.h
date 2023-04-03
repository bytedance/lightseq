#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include "kernels.h"
#include <stdexcept>

namespace lightseq {
namespace cuda {

/**
@brief: launch_gpt_embedding
for gpt embedding, look up token embedding, add position embedding

@param
token_emb: [vocab_size, hidden_size]
pos_emb: [max_step, hidden_size]
token_id: input token id, [batch_size, beam_size, max_step]
output: result, [batch_size, token_seq_len, hidden_size]
padding_id, the padding_id, default 0
pos_offset: get real pos when decoding which gridDim.y=1
*/

template <typename T>
void launch_gpt_embedding(const T* token_emb, const T* pos_emb,
                          const int* tokens, T* output, T* pad_mask_ptr,
                          int* left_pad_len_ptr, int batch_size, int beam_size,
                          int hidden_dim, int step_offset, int seq_len,
                          int max_step, int padding_id, cudaStream_t stream);

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
