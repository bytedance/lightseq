#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <vector>
#include <cub/cub.cuh>

#define MAX_THREADS 1024
#define WARP_SIZE 32

namespace lightseq {
namespace cuda {
template <typename T>
void launch_process_logits(const int* enc_input_ids, const int* pad_mask,
                           const int* dec_prev_ids, int8_t* banned_tokens,
                           int enc_seq_len, int cur_step, int max_step,
                           int encoder_no_repeat_ngram_size,
                           int no_repeat_ngram_size, int vocab_size,
                           int batch_size, int beam_size, cudaStream_t stream);
template <typename T>
void masked_select_beam_rough_topk_launcher(
    const T* logits, const T* logit_bias, const float* seq_probs,
    const float* seq_score, const int* alive_seq, int* can_idx,
    float* can_score, int* num_beam_can, int8_t* banned_tokens, int vocab_size,
    int max_step, float length_norm, int cur_step, int step_token_num,
    int max_thread_per_block, cudaStream_t stream, int beam_size,
    float diverse_lambda, int end_id);
}  // namespace cuda
}  // namespace lightseq
