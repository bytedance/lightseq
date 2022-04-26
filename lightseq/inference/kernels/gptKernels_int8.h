#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>

namespace lightseq {
namespace cuda {

template <typename T>
void ker_gpt_embedding_i8I_launcher(int batch_size, int batch_seq_len,
                                    int hidden_size, cudaStream_t stream,
                                    const int8_t* token_emb, const T* pos_emb,
                                    const int* token_id, T* output,
                                    int* real_seq_len, int padding_id,
                                    int pos_offset, float dequant_scale);

void ker_ppl_i8I_launcher(int batch_size, int batch_seq_len,
                          int max_thread_per_block, cudaStream_t stream,
                          const int8_t* logits, const int* input_ids,
                          const int* real_seq_len, float* ppl, int vocab_size,
                          float dequant_scale, bool in_col32 = false);

template <typename T>
void ker_correlation_softmax_gpt_i32I_launcher(
    int batch_size, int batch_seq_len, int head_num, cudaStream_t stream,
    int32_t* correlation, T* output, const int* real_seq_len, float attn_scale,
    float dequant_scale);

}  // namespace cuda
}  // namespace lightseq
