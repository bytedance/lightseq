#include "common.h"
#include "generation.h"
#include "transformerKernels.h"

namespace lightseq {
namespace cuda {
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
banned_tokens: [batch_size, beam_size, vocab_size]
encoder_no_repeat_ngram_size < MAX_THREADS
no_repeat_ngram_size < MAX_THREADS
*/
template <typename T>
__global__ void ker_process_logits(const int* enc_input_ids,
                                   const int* pad_mask, const int* dec_prev_ids,
                                   int8_t* banned_tokens, int enc_seq_len,
                                   int cur_step, int max_step,
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
      if (pad_mask[flat_2dim(blockIdx.x, i + n_match, enc_seq_len)] == 1) {
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
        banned_tokens[flat_3dim(blockIdx.x, blockIdx.y, banned_token, beam_size,
                                vocab_size)] = -1;
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
        banned_tokens[flat_3dim(blockIdx.x, blockIdx.y, banned_token, beam_size,
                                vocab_size)] = -1;
      }
    }
    __syncthreads();
  }

  for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
    int8_t* trg = banned_tokens;
    trg += flat_3dim(blockIdx.x, blockIdx.y, i, beam_size, vocab_size);
    if (*trg == -1) {
      *trg = 1;
    } else {
      *trg = 0;
    }
  }
}

template <typename T>
void launch_process_logits(const int* enc_input_ids, const int* pad_mask,
                           const int* dec_prev_ids, int8_t* banned_tokens,
                           int enc_seq_len, int cur_step, int max_step,
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
      enc_input_ids, pad_mask, dec_prev_ids, banned_tokens, enc_seq_len,
      cur_step, max_step, encoder_no_repeat_ngram_size, no_repeat_ngram_size,
      vocab_size);
}

template void launch_process_logits<float>(
    const int* enc_input_ids, const int* pad_mask, const int* dec_prev_ids,
    int8_t* banned_tokens, int enc_seq_len, int cur_step, int max_step,
    int encoder_no_repeat_ngram_size, int no_repeat_ngram_size, int vocab_size,
    int batch_size, int beam_size, cudaStream_t stream);

template void launch_process_logits<__half>(
    const int* enc_input_ids, const int* pad_mask, const int* dec_prev_ids,
    int8_t* banned_tokens, int enc_seq_len, int cur_step, int max_step,
    int encoder_no_repeat_ngram_size, int no_repeat_ngram_size, int vocab_size,
    int batch_size, int beam_size, cudaStream_t stream);

/**
@brief: masked_select_beam_rough_topk
one block for one beam, compute the log seq probability ended with every token
in
vocab, base on the previous log seq probability and current step's logit, select
rough topK candidate.

@thread
gridDim.x = batch_size * beam_size
blockDim.x = max_thread_per_block

@param
logits: [batch_size, beam_size, vocab_size], cur step logit
logit_bias: [vocab_size], logit bias
seq_probs: [batch_size, beam_size], prefix sequence log probability
seq_score: [batch_size, beam_size], prefix sequence score
alive_seq: [batch_size, beam_size, max_step], prefix sequence id
can_idx: [batch_size, beam_size, vocab_size], topk candidate's index
can_score: [batch_size, beam_size, vocab_size], topk candidate's score
num_beam_can: [1 + batch_size * beam_size].
    the first ele save the number of topk candidate of the whole batch
    the remaining batch_size * beam_size ele save the number of topk candidate
    of each beam
vocab_size: the vocab size of decoder
max_step: max decode step
length_norm: length penlty value for current step
cur_step: current step
diverse_lambda: lambda for diverse beam search
*/
template <typename T, int beam_size>
__global__ void masked_select_beam_rough_topk(
    const T* logits, const T* logit_bias, const float* seq_probs,
    const float* seq_score, const int* alive_seq, int* can_idx,
    float* can_score, int* num_beam_can, int8_t* banned_tokens, int vocab_size,
    int max_step, float length_norm, int cur_step, float diverse_lambda,
    int end_id) {
  if (cur_step != 0 && alive_seq[blockIdx.x * max_step + cur_step] == end_id) {
    // this is a finished beam
    if (threadIdx.x == 0) {
      num_beam_can[blockIdx.x + 1] = 1;      // generate one candidate
      int pos = atomicAdd(num_beam_can, 1);  // get a candidate pos
      if (diverse_lambda == 0) {
        can_score[pos] =
            seq_score[blockIdx.x];  // this beam's score will not be change
      } else {
        // add the beam id offset in score to sort in each beam
        int batch_id = blockIdx.x / beam_size;
        can_score[pos] = seq_score[blockIdx.x] +
                         (blockIdx.x - batch_id) * min_log_probability;
      }
      can_idx[pos] = end_id + (blockIdx.x % beam_size) * vocab_size;  // EOS
    }
    return;
  }

  /* step1: compute each thread's max_logit and sum_exp_logit, store in
   * rough_top_kth_logit, sum_exp_logit */
  const int block_start = blockIdx.x * vocab_size;
  const int left_idx = block_start + threadIdx.x;
  const int right_idx = (blockIdx.x + 1) * vocab_size;
  float rough_top_kth_logit = CUDA_FLOAT_INF_NEG;
  float sum_exp_logit = 0;
  for (int i = left_idx; i < right_idx; i += blockDim.x) {
    if (banned_tokens[i] == 0) {
      float lgt = (float)logits[i] + (float)__ldg(&logit_bias[i - block_start]);
      rough_top_kth_logit = fmaxf(rough_top_kth_logit, lgt);
    }
  }
  float max_logit = blockReduceMax(rough_top_kth_logit);
  __shared__ float s_max_logit;
  if (threadIdx.x == 0) {
    s_max_logit = max_logit;
  }
  __syncthreads();
  for (int i = left_idx; i < right_idx; i += blockDim.x) {
    float lgt =
        fmaxf((float)(logits[i]) + (float)__ldg(&logit_bias[i - block_start]) -
                  s_max_logit,
              logit_thresh_min);
    sum_exp_logit += expf(lgt);
  }

  /*
  step2: compute rough top-kth-logits and sum_exp_logit among the whole beam,
  saved into s_topk and
      s_log_prob_base
  */
  __shared__ float
      s_log_prob_base;      // prefix sequence log prob - log_sum_exp_logit
  __shared__ float s_topk;  // rough top k-th value of logits
  __shared__ int num_cur_beam_can;  // candidate number for this beam
  sum_exp_logit = blockReduceSum(sum_exp_logit);
  rough_top_kth_logit = blockRoughTopK<float, beam_size>(rough_top_kth_logit);
  if (threadIdx.x == 0) {
    s_log_prob_base = seq_probs[blockIdx.x] - logf(sum_exp_logit) - s_max_logit;
    s_topk = rough_top_kth_logit;
    num_cur_beam_can = 0;
  }

  /*
  step3 : select the candidate token with logits bigger than s_topk,
          compute the seq probability ended with them,
      save the probability, token_index, selected token number.
  */
  int idx = left_idx;
  int batch_id = blockIdx.x / beam_size;
  int batch_start_pos = batch_id * beam_size * vocab_size;
  // int unk_vocab_id = vocab_size - 3;  // last three element: unk, start, eos
  __shared__ int l_n;  // current iteration candidate number
  for (int iter = 0; iter < (vocab_size + blockDim.x - 1) / blockDim.x;
       iter++) {
    // zero the counter
    if (threadIdx.x == 0) l_n = 0;
    __syncthreads();

    float lgt = CUDA_FLOAT_INF_NEG - 1.f;  // min s_topk is CUDA_FLOAT_INF_NEG
    int pos;
    int vocab_id = idx - block_start;

    // if ((vocab_id < vocab_size) && (vocab_id != unk_vocab_id)) {
    if (vocab_id < vocab_size && banned_tokens[idx] == 0) {
      lgt = (float)(logits[idx]) + (float)__ldg(&logit_bias[vocab_id]);
      if (lgt >= s_topk)
        // pos: relative pos inside this iteration
        pos = atomicAdd(&l_n, 1);
    }
    __syncthreads();

    // leader increments the global counter
    if (threadIdx.x == 0) {
      atomicAdd(&num_cur_beam_can, l_n);
      l_n = atomicAdd(num_beam_can, l_n);
    }
    __syncthreads();

    // threads with true predicates write their elements
    if ((lgt >= s_topk)) {
      pos += l_n;  // increment local pos by global counter
      if (diverse_lambda == 0) {
        can_score[pos] = fmaxf((lgt + s_log_prob_base) * length_norm,
                               min_log_probability + 1.f) +
                         batch_id * min_log_probability;
      } else {
        can_score[pos] = fmaxf((lgt + s_log_prob_base) * length_norm,
                               min_log_probability + 1.f) +
                         blockIdx.x * min_log_probability;
      }
      can_idx[pos] = idx - batch_start_pos;
    }
    __syncthreads();
    idx += blockDim.x;
  }
  if (threadIdx.x == 0) {
    num_beam_can[blockIdx.x + 1] = num_cur_beam_can;
  }
}

template <typename T>
void masked_select_beam_rough_topk_launcher(
    const T* logits, const T* logit_bias, const float* seq_probs,
    const float* seq_score, const int* alive_seq, int* can_idx,
    float* can_score, int* num_beam_can, int8_t* banned_tokens, int vocab_size,
    int max_step, float length_norm, int cur_step, int step_token_num,
    int max_thread_per_block, cudaStream_t stream, int beam_size,
    float diverse_lambda, int end_id) {
  if (beam_size == 1)
    masked_select_beam_rough_topk<T, 1>
        <<<step_token_num, max_thread_per_block, 0, stream>>>(
            logits, logit_bias, seq_probs, seq_score, alive_seq, can_idx,
            can_score, num_beam_can, banned_tokens, vocab_size, max_step,
            length_norm, cur_step, diverse_lambda, end_id);
  if (beam_size == 2)
    masked_select_beam_rough_topk<T, 2>
        <<<step_token_num, max_thread_per_block, 0, stream>>>(
            logits, logit_bias, seq_probs, seq_score, alive_seq, can_idx,
            can_score, num_beam_can, banned_tokens, vocab_size, max_step,
            length_norm, cur_step, diverse_lambda, end_id);
  if (beam_size == 4)
    masked_select_beam_rough_topk<T, 4>
        <<<step_token_num, max_thread_per_block, 0, stream>>>(
            logits, logit_bias, seq_probs, seq_score, alive_seq, can_idx,
            can_score, num_beam_can, banned_tokens, vocab_size, max_step,
            length_norm, cur_step, diverse_lambda, end_id);
  if (beam_size == 8)
    masked_select_beam_rough_topk<T, 8>
        <<<step_token_num, max_thread_per_block, 0, stream>>>(
            logits, logit_bias, seq_probs, seq_score, alive_seq, can_idx,
            can_score, num_beam_can, banned_tokens, vocab_size, max_step,
            length_norm, cur_step, diverse_lambda, end_id);
  if (beam_size == 16)
    masked_select_beam_rough_topk<T, 16>
        <<<step_token_num, max_thread_per_block, 0, stream>>>(
            logits, logit_bias, seq_probs, seq_score, alive_seq, can_idx,
            can_score, num_beam_can, banned_tokens, vocab_size, max_step,
            length_norm, cur_step, diverse_lambda, end_id);
  if (beam_size == 32)
    masked_select_beam_rough_topk<T, 32>
        <<<step_token_num, max_thread_per_block, 0, stream>>>(
            logits, logit_bias, seq_probs, seq_score, alive_seq, can_idx,
            can_score, num_beam_can, banned_tokens, vocab_size, max_step,
            length_norm, cur_step, diverse_lambda, end_id);
}

template void masked_select_beam_rough_topk_launcher<float>(
    const float* logits, const float* logit_bias, const float* seq_probs,
    const float* seq_score, const int* alive_seq, int* can_idx,
    float* can_score, int* num_beam_can, int8_t* banned_tokens, int vocab_size,
    int max_step, float length_norm, int cur_step, int step_token_num,
    int max_thread_per_block, cudaStream_t stream, int beam_size,
    float diverse_lambda, int end_id);

template void masked_select_beam_rough_topk_launcher<__half>(
    const __half* logits, const __half* logit_bias, const float* seq_probs,
    const float* seq_score, const int* alive_seq, int* can_idx,
    float* can_score, int* num_beam_can, int8_t* banned_tokens, int vocab_size,
    int max_step, float length_norm, int cur_step, int step_token_num,
    int max_thread_per_block, cudaStream_t stream, int beam_size,
    float diverse_lambda, int end_id);
}  // namespace cuda
}  // namespace lightseq
