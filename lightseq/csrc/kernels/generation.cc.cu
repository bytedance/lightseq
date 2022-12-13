#include "common.h"
#include "kernels.h"
// namespace lightseq {
// namespace cuda {
/**
@brief: ker_process_logits
process logits before generation, seee
https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py
currently only support blocking ngram which shows in encoder and decoder prefix

@thread
gridDim.x = batch_size
gridDim.y = beam_size
blockDim.x = MAX_THREADS

@param
enc_input_ids: [batch_size, enc_seq_len]
pad_mask: [batch_size, enc_seq_len], -inf for padding token, 0.0 for others
dec_prev_ids: [batch_size, beam_size, max_step]
logits: [batch_size, beam_size, vocab_size]
encoder_no_repeat_ngram_size < MAX_THREADS
no_repeat_ngram_size < MAX_THREADS
*/
template <typename T>
__global__ void ker_process_logits(const int* enc_input_ids, const T* pad_mask,
                                   const int* dec_prev_ids, T* logits,
                                   int enc_seq_len, int cur_step, int max_step,
                                   int encoder_no_repeat_ngram_size,
                                   int no_repeat_ngram_size, int vocab_size) {
  // max(encoder_no_repeat_ngram_size, no_repeat_ngram_size)
  extern __shared__ int smen[];
  int beam_size = gridDim.y;

  // block ngram in encoder
  if (encoder_no_repeat_ngram_size > 0 &&
      cur_step >= encoder_no_repeat_ngram_size - 1) {
    int n_match = encoder_no_repeat_ngram_size - 1;
    if (threadIdx.x < n_match) {
      smen[threadIdx.x] = dec_prev_ids[flat_3dim(
          blockIdx.x, blockIdx.y, cur_step - n_match + threadIdx.x, beam_size,
          max_step)];
    }
    __syncthreads();
    for (int i = threadIdx.x; i <= enc_seq_len - 1 - n_match; i += blockDim.x) {
      if (pad_mask[flat_2dim(blockIdx.x, i + n_match, enc_seq_len)] < (T)0) {
        break;  // ngram's end is padding tokens
      }
      int j = 0;
      for (j = 0; j < n_match; j++) {
        if (enc_input_ids[flat_2dim(blockIdx.x, i + j, enc_seq_len)] !=
            smen[j]) {
          break;
        }
      }
      if (j == n_match) {
        int banned_token =
            enc_input_ids[flat_2dim(blockIdx.x, i + j, enc_seq_len)];
        T* trg = logits + flat_3dim(blockIdx.x, blockIdx.y, banned_token,
                                    beam_size, vocab_size);
        atomicAdd(trg, (T)lightseq::CUDA_FLOAT_INF_NEG);
      }
    }
    __syncthreads();
  }

  // block ngram in decoder
  if (no_repeat_ngram_size > 0 && cur_step >= no_repeat_ngram_size - 1) {
    int n_match = no_repeat_ngram_size - 1;
    if (threadIdx.x < n_match) {
      smen[threadIdx.x] = dec_prev_ids[flat_3dim(
          blockIdx.x, blockIdx.y, cur_step - n_match + threadIdx.x, beam_size,
          max_step)];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < cur_step - no_repeat_ngram_size + 1;
         i += blockDim.x) {
      int j = 0;
      for (j = 0; j < n_match; j++) {
        if (dec_prev_ids[flat_3dim(blockIdx.x, blockIdx.y, i + j, beam_size,
                                   max_step)] != smen[j]) {
          break;
        }
      }
      if (j == n_match) {
        int banned_token = dec_prev_ids[flat_3dim(blockIdx.x, blockIdx.y, i + j,
                                                  beam_size, max_step)];
        T* trg = logits + flat_3dim(blockIdx.x, blockIdx.y, banned_token,
                                    beam_size, vocab_size);
        *trg = (T)lightseq::CUDA_FLOAT_INF_NEG;
      }
    }
    __syncthreads();
  }
}

template <typename T>
void launch_process_logits(const int* enc_input_ids, const T* pad_mask,
                           const int* dec_prev_ids, T* logits, int enc_seq_len,
                           int cur_step, int max_step,
                           int encoder_no_repeat_ngram_size,
                           int no_repeat_ngram_size, int vocab_size,
                           int batch_size, int beam_size, cudaStream_t stream) {
  if (no_repeat_ngram_size > MAX_THREADS ||
      encoder_no_repeat_ngram_size > MAX_THREADS) {
    throw std::runtime_error(
        "no_repeat_ngram_size exceeds the maximum (1024)!");
  }

  dim3 grid_dim(batch_size, beam_size);
  dim3 block_dim(MAX_THREADS);
  int smen_byte_size =
      max(encoder_no_repeat_ngram_size, no_repeat_ngram_size) * sizeof(int);
  ker_process_logits<T><<<grid_dim, block_dim, smen_byte_size, stream>>>(
      enc_input_ids, pad_mask, dec_prev_ids, logits, enc_seq_len, cur_step,
      max_step, encoder_no_repeat_ngram_size, no_repeat_ngram_size, vocab_size);
}

template void launch_process_logits<float>(
    const int* enc_input_ids, const float* pad_mask, const int* dec_prev_ids,
    float* logits, int enc_seq_len, int cur_step, int max_step,
    int encoder_no_repeat_ngram_size, int no_repeat_ngram_size, int vocab_size,
    int batch_size, int beam_size, cudaStream_t stream);

template void launch_process_logits<__half>(
    const int* enc_input_ids, const __half* pad_mask, const int* dec_prev_ids,
    __half* logits, int enc_seq_len, int cur_step, int max_step,
    int encoder_no_repeat_ngram_size, int no_repeat_ngram_size, int vocab_size,
    int batch_size, int beam_size, cudaStream_t stream);

//}
//}
