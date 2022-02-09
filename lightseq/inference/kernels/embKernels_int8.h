#pragma once
#include <cuda.h>
#include <cuda_fp16.h>

namespace lightseq {
namespace cuda {

template <typename T>
void launch_enc_emb_i8I(const int8_t *token_emb, const T *pos_emb,
                        const int *tokens, T *output, int *pad_mask, int pad_id,
                        int batch_size, int seq_len, int hidden_dim,
                        cudaStream_t stream, const T *lang_emb,
                        const int *lang_id, int multilg_type,
                        float dequant_scale);

template <typename T>
void launch_dec_emb_i8I(const int8_t *token_emb, const T *pos_emb, int *tokens,
                        const T *lang_emb, const int *lang_id, T *output,
                        int batch_size, int beam_size, int hidden_dim,
                        int vocab_size, int step, int max_step,
                        int multilg_type, cudaStream_t stream,
                        float dequant_scale);

}  // namespace cuda
}  // namespace lightseq
