#pragma once
#include <cuda.h>
#include <cuda_fp16.h>

namespace byseqlib {
namespace cuda {

template <typename T>
void ker_gpt_embedding_launcher(int batch_size, int batch_seq_len,
                                int hidden_size, cudaStream_t stream,
                                const T* token_emb, const T* pos_emb,
                                const int* token_id, T* output,
                                int* real_seq_len, int padding_id);

template <typename T>
void ker_bias_gelu_launcher(int batch_token_num, int block_dim,
                            cudaStream_t stream, T* input, const T* bias,
                            int feature_dim);

template <typename T>
void ker_correlation_softmax_gpt_launcher(int batch_size, int batch_seq_len,
                                          int head_num, cudaStream_t stream,
                                          T* correlation,
                                          const int* real_seq_len);

template <typename T>
void ker_ppl_launcher(int batch_size, int batch_seq_len,
                      int max_thread_per_block, cudaStream_t stream,
                      const T* logits, const int* input_ids,
                      const int* real_seq_len, float* ppl, int vocab_size);

}  // namespace cuda
}  // namespace byseqlib
