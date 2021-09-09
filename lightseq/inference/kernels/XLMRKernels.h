#pragma once
#include <cuda.h>
#include <cuda_fp16.h>

namespace lightseq {
namespace cuda {

template <typename T>
void ker_xlmr_enc_emb_launcher(int batch_size, int batch_seq_len,
                               int hidden_size, cudaStream_t stream,
                               const T* token_emb, const T* pos_emb,
                               const T* src_lang_emb, const int* token_id,
                               const int* lang_id, T* output, int* padding_mask,
                               int padding_id, int max_thread_per_block);

template <typename T>
void ker_xlmr_dec_emb_launcher(int step_token_num, int hidden_size,
                               cudaStream_t stream, const T* token_emb,
                               const T* pos_emb, const T* trg_lang_emb,
                               const int* token_id, const int* lang_id,
                               T* output, int step, int max_step,
                               int vocab_size, int beam_size,
                               int max_thread_per_block);

}  // namespace cuda
}  // namespace lightseq
