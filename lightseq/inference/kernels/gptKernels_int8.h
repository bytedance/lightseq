#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>

namespace lightseq {
namespace cuda {

template <typename T>
void ker_gpt_embedding_int8_launcher(int batch_size, int batch_seq_len,
                                int hidden_size, cudaStream_t stream,
                                const int8_t* token_emb, const T* pos_emb,
                                const int* token_id, T* output,
                                int* real_seq_len, int padding_id,
                                int pos_offset, float dequant_scale);

}  // namespace cuda
}  // namespace lightseq
