#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <vector>

#define MAX_THREADS 1024
#define WARP_SIZE 32

const float kQuantRangeI8 = 127.0f;

namespace lightseq {
namespace cuda {
template <typename T>
void launch_process_logits(const int *enc_input_ids, const int *pad_mask,
                           const int *dec_prev_ids, T *logits, int enc_seq_len,
                           int cur_step, int max_step,
                           int encoder_no_repeat_ngram_size,
                           int no_repeat_ngram_size, int vocab_size,
                           int batch_size, int beam_size, cudaStream_t stream);
} // namespace cuda
} // namespace lightseq