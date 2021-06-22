#pragma once
#include <cuda.h>
#include <cuda_fp16.h>

namespace lightseq {
namespace cuda {

template <typename T>
void ker_multilg_enc_emb_launcher(int batch_size, int batch_seq_len,
                                  int hidden_size, cudaStream_t stream,
                                  const T* token_emb, const T* pos_emb,
                                  const T* src_lang_emb, const int* token_id,
                                  T* output, int* padding_mask, int padding_id,
                                  int max_thread_per_block);

template <typename T>
void ker_multilg_dec_emb_launcher(int step_token_num, int hidden_size,
                                  cudaStream_t stream, const T* token_emb,
                                  const T* pos_emb, const T* src_lang_emb,
                                  const T* trg_lang_emb,
                                  const int* src_token_id, const int* token_id,
                                  T* output, int step, int max_step,
                                  int vocab_size, int beam_size,
                                  int src_seq_len, int max_thread_per_block);

template <typename T>
void select_beam_rough_topk_multilg_launcher(
    const T* logits, const T* logit_bias, const float* seq_probs,
    const float* seq_score, const int* alive_seq, const int* vocab_mask,
    const int* src_token_id, int* can_idx, float* can_score, int* num_beam_can,
    int vocab_size, int max_step, float length_norm, int cur_step,
    int step_token_num, int max_thread_per_block, cudaStream_t stream,
    int beam_size, float diverse_lambda, int end_id, int src_seq_len);

}  // namespace cuda
}  // namespace lightseq
