#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>

namespace lightseq {
namespace cuda {

    const float t5_epsilon = 1e-6;
    template <typename T>
    void t5_ker_norm_layer_launcher(int token_num, int hidden_size,
                                cudaStream_t stream, T* matrix, T* out, const T* scale,
                                const T* bias, int max_thread_per_block);

    template <typename T>
    void t5_ker_correlation_softmax_encself_launcher(int batch_size, int batch_seq_len,
                                                int head_num, cudaStream_t stream,
                                                T* correlation,
                                                const int* src_padding_mask,
                                                const T *pos_emb);

    template <typename T>
    void t5_ker_correlation_softmax_decself_launcher(int batch_head_num, int step_num,
                                                cudaStream_t stream,
                                                T* correlation,
                                                const T *pos_emb, int head_num);
}  // namespace cuda
}  // namespace lightseq
