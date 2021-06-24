#include <random>

#include "common.h"
#include "multilgKernels.h"
#include "transformerKernels.h"
/**
@file
Implemented the cuda kernel function and its launcher
that required by multilingual nmt model.
Currently, fp16 and fp32 versions are provided
*/
namespace lightseq {
namespace cuda {
/**
@brief: ker_multilg_enc_emb
for encoder, look up token embedding, add position embedding

@thread
gridDim.x = batch_size
gridDim.y = batch_seq_len
blockDim.x = max_thread_per_block

@param
token_emb: [vocab_size, hidden_size]
pos_emb: [max_step, hidden_size]
token_id: input token id, [batch_size, batch_seq_len]
output: result, [batch_size, batch_seq_len, hidden_size]
padding_mask: record the padding token, [batch_size, batch_seq_len]
padding_id, the padding token id
*/
template <typename T>
__global__ void ker_multilg_enc_emb(const T* token_emb, const T* pos_emb,
                                    const T* src_lang_emb, const int* token_id,
                                    T* output, int* padding_mask,
                                    int padding_id, const int hidden_size) {
  int target_pos = blockIdx.x * gridDim.y + blockIdx.y;
  int start = target_pos * hidden_size + threadIdx.x;
  int end = (target_pos + 1) * hidden_size;
  int tid = token_id[target_pos];
  int lang_id = token_id[blockIdx.x * gridDim.y];
  if (tid == padding_id) {
    // for padding id
    if (threadIdx.x == 0) padding_mask[target_pos] = 1;
    for (uint i = start; i < end; i += blockDim.x) {
      // output[target_pos * blockDim.x + threadIdx.x] = 0.f;
      output[i] = 0.f;
    }
    return;
  }
  if (threadIdx.x == 0) {
    padding_mask[target_pos] = 0;
  }
  for (uint i = start; i < end; i += blockDim.x) {
    int offset = i - target_pos * hidden_size;
    output[i] = token_emb[tid * hidden_size + offset] +
                pos_emb[blockIdx.y * hidden_size + offset] +
                src_lang_emb[lang_id * hidden_size + offset];
  }
}

template <>
__global__ void ker_multilg_enc_emb<__half>(const __half* token_emb,
                                            const __half* pos_emb,
                                            const __half* src_lang_emb,
                                            const int* token_id, __half* output,
                                            int* padding_mask, int padding_id,
                                            const int half_hidden_size) {
  int target_pos = blockIdx.x * gridDim.y + blockIdx.y;
  int start = target_pos * half_hidden_size + threadIdx.x;
  int end = (target_pos + 1) * half_hidden_size;
  int tid = token_id[target_pos];
  int lang_id = token_id[blockIdx.x * gridDim.y];
  half2* output_h = (half2*)output;

  if (tid == padding_id) {
    // for padding id
    if (threadIdx.x == 0) padding_mask[target_pos] = 1;
    for (uint i = start; i < end; i += blockDim.x) {
      output_h[i] = __float2half2_rn(0.f);
    }
    return;
  }
  if (threadIdx.x == 0) {
    padding_mask[target_pos] = 0;
  }
  for (uint i = start; i < end; i += blockDim.x) {
    int offset = i - target_pos * half_hidden_size;
    float2 te = __half22float2(
        ((const half2*)token_emb)[tid * half_hidden_size + offset]);
    float2 pe = __half22float2(
        ((const half2*)pos_emb)[blockIdx.y * half_hidden_size + offset]);
    float2 le = __half22float2(
        ((const half2*)src_lang_emb)[lang_id * half_hidden_size + offset]);
    te.x = te.x + pe.x + le.x;
    te.y = te.y + pe.y + le.y;

    output_h[i] = __float22half2_rn(te);
  }
}

template <typename T>
void ker_multilg_enc_emb_launcher(int batch_size, int batch_seq_len,
                                  int hidden_size, cudaStream_t stream,
                                  const T* token_emb, const T* pos_emb,
                                  const T* src_lang_emb, const int* token_id,
                                  T* output, int* padding_mask, int padding_id,
                                  int max_thread_per_block) {
  ker_multilg_enc_emb<T>
      <<<dim3(batch_size, batch_seq_len), max_thread_per_block, 0, stream>>>(
          token_emb, pos_emb, src_lang_emb, token_id, output, padding_mask,
          padding_id, hidden_size);
}

template <>
void ker_multilg_enc_emb_launcher<__half>(
    int batch_size, int batch_seq_len, int hidden_size, cudaStream_t stream,
    const __half* token_emb, const __half* pos_emb, const __half* src_lang_emb,
    const int* token_id, __half* output, int* padding_mask, int padding_id,
    int max_thread_per_block) {
  ker_multilg_enc_emb<__half>
      <<<dim3(batch_size, batch_seq_len), max_thread_per_block, 0, stream>>>(
          token_emb, pos_emb, src_lang_emb, token_id, output, padding_mask,
          padding_id, hidden_size / 2);
}

template void ker_multilg_enc_emb_launcher<float>(
    int batch_size, int batch_seq_len, int hidden_size, cudaStream_t stream,
    const float* token_emb, const float* pos_emb, const float* src_lang_emb,
    const int* token_id, float* output, int* padding_mask, int padding_id,
    int max_thread_per_block);

template void ker_multilg_enc_emb_launcher<__half>(
    int batch_size, int batch_seq_len, int hidden_size, cudaStream_t stream,
    const __half* token_emb, const __half* pos_emb, const __half* src_lang_emb,
    const int* token_id, __half* output, int* padding_mask, int padding_id,
    int max_thread_per_block);

/**
@brief: ker_multilg_dec_emb
for multilingual decoder, look up token embedding, add position embedding
and lang embedding

@thread
gridDim.x = batch_size * beam_size
blockDim.x = max_thread_per_block

@param
token_emb: [hidden_size, vocab_size], note, it is different with encoder
pos_emb: [max_step, hidden_size]
src_lang_emb: [lang_num, hidden_size]
trg_lang_emb: [lang_num, hidden_size]
src_token_id: [batch_size, src_seq_len]
token_id: input token id, [batch_size, beam_size, max_step]
output: result, [batch_size, beam_size, hidden_size]
step: current step
max_step: max decoder steps
vocab_size: vocabulary size
*/
template <typename T>
__global__ void ker_multilg_dec_emb(
    const T* token_emb, const T* pos_emb, const T* src_lang_emb,
    const T* trg_lang_emb, const int* src_token_id, const int* token_id,
    T* output, int step, int max_step, int vocab_size, int hidden_size,
    int beam_size, int src_seq_len) {
  int batch_id = blockIdx.x / beam_size;
  // src seq is in [src_lang_id, trg_lang_id, tokens...] format
  int src_lang_id = src_token_id[batch_id * src_seq_len];
  int trg_lang_id = src_token_id[batch_id * src_seq_len + 1];
  int token_idx =
      (step == 0 ? trg_lang_id : token_id[blockIdx.x * max_step + step]);
  for (uint offset = threadIdx.x; offset < hidden_size; offset += blockDim.x) {
    output[blockIdx.x * hidden_size + offset] =
        token_emb[offset * vocab_size + token_idx] +
        pos_emb[step * hidden_size + offset] +
        src_lang_emb[src_lang_id * hidden_size + offset] +
        trg_lang_emb[trg_lang_id * hidden_size + offset];
  }
}

template <typename T>
void ker_multilg_dec_emb_launcher(int step_token_num, int hidden_size,
                                  cudaStream_t stream, const T* token_emb,
                                  const T* pos_emb, const T* src_lang_emb,
                                  const T* trg_lang_emb,
                                  const int* src_token_id, const int* token_id,
                                  T* output, int step, int max_step,
                                  int vocab_size, int beam_size,
                                  int src_seq_len, int max_thread_per_block) {
  ker_multilg_dec_emb<T><<<step_token_num, max_thread_per_block, 0, stream>>>(
      token_emb, pos_emb, src_lang_emb, trg_lang_emb, src_token_id, token_id,
      output, step, max_step, vocab_size, hidden_size, beam_size, src_seq_len);
}

template void ker_multilg_dec_emb_launcher<float>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const float* token_emb, const float* pos_emb, const float* src_lang_emb,
    const float* trg_lang_emb, const int* src_token_id, const int* token_id,
    float* output, int step, int max_step, int vocab_size, int beam_size,
    int src_seq_len, int max_thread_per_block);

template void ker_multilg_dec_emb_launcher<__half>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const __half* token_emb, const __half* pos_emb, const __half* src_lang_emb,
    const __half* trg_lang_emb, const int* src_token_id, const int* token_id,
    __half* output, int step, int max_step, int vocab_size, int beam_size,
    int src_seq_len, int max_thread_per_block);

/**
@brief: select_beam_rough_topk_multilg
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
__global__ void select_beam_rough_topk_multilg(
    const T* logits, const T* logit_bias, const float* seq_probs,
    const float* seq_score, const int* alive_seq, const int* vocab_mask,
    const int* src_token_id, int* can_idx, float* can_score, int* num_beam_can,
    int vocab_size, int max_step, float length_norm, int cur_step,
    float diverse_lambda, int end_id, int src_seq_len) {
  if (alive_seq[blockIdx.x * max_step + cur_step] == end_id) {
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
  int batch_id = blockIdx.x / beam_size;
  int trg_lang_id = src_token_id[batch_id * src_seq_len + 1];
  const int block_start = blockIdx.x * vocab_size;
  const int left_idx = block_start + threadIdx.x;
  const int right_idx = (blockIdx.x + 1) * vocab_size;
  float rough_top_kth_logit = CUDA_FLOAT_INF_NEG;
  float sum_exp_logit = 0;
  for (int i = left_idx; i < right_idx; i += blockDim.x) {
    int lang_mask = vocab_mask[trg_lang_id * vocab_size + i - block_start];
    float lgt =
        (lang_mask == 0
             ? CUDA_FLOAT_INF_NEG
             : (float)logits[i] + (float)__ldg(&logit_bias[i - block_start]));
    rough_top_kth_logit = fmaxf(rough_top_kth_logit, lgt);
  }
  float max_logit = blockReduceMax(rough_top_kth_logit);
  __shared__ float s_max_logit;
  if (threadIdx.x == 0) {
    s_max_logit = max_logit;
  }
  __syncthreads();
  for (int i = left_idx; i < right_idx; i += blockDim.x) {
    int lang_mask = vocab_mask[trg_lang_id * vocab_size + i - block_start];
    float lgt =
        lang_mask == 0
            ? 0.f
            : expf(fmaxf((float)(logits[i]) +
                             (float)__ldg(&logit_bias[i - block_start]) -
                             s_max_logit,
                         logit_thresh_min));
    sum_exp_logit += lgt;
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
    if (vocab_id < vocab_size) {
      int lang_mask = vocab_mask[trg_lang_id * vocab_size + vocab_id];
      if (lang_mask != 0) {
        lgt = (float)(logits[idx]) + (float)__ldg(&logit_bias[vocab_id]);
        if (lgt >= s_topk)
          // pos: relative pos inside this iteration
          pos = atomicAdd(&l_n, 1);
      }
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
void select_beam_rough_topk_multilg_launcher(
    const T* logits, const T* logit_bias, const float* seq_probs,
    const float* seq_score, const int* alive_seq, const int* vocab_mask,
    const int* src_token_id, int* can_idx, float* can_score, int* num_beam_can,
    int vocab_size, int max_step, float length_norm, int cur_step,
    int step_token_num, int max_thread_per_block, cudaStream_t stream,
    int beam_size, float diverse_lambda, int end_id, int src_seq_len) {
  if (beam_size == 1)
    select_beam_rough_topk_multilg<T, 1>
        <<<step_token_num, max_thread_per_block, 0, stream>>>(
            logits, logit_bias, seq_probs, seq_score, alive_seq, vocab_mask,
            src_token_id, can_idx, can_score, num_beam_can, vocab_size,
            max_step, length_norm, cur_step, diverse_lambda, end_id,
            src_seq_len);
  if (beam_size == 2)
    select_beam_rough_topk_multilg<T, 2>
        <<<step_token_num, max_thread_per_block, 0, stream>>>(
            logits, logit_bias, seq_probs, seq_score, alive_seq, vocab_mask,
            src_token_id, can_idx, can_score, num_beam_can, vocab_size,
            max_step, length_norm, cur_step, diverse_lambda, end_id,
            src_seq_len);
  if (beam_size == 4)
    select_beam_rough_topk_multilg<T, 4>
        <<<step_token_num, max_thread_per_block, 0, stream>>>(
            logits, logit_bias, seq_probs, seq_score, alive_seq, vocab_mask,
            src_token_id, can_idx, can_score, num_beam_can, vocab_size,
            max_step, length_norm, cur_step, diverse_lambda, end_id,
            src_seq_len);
  if (beam_size == 8)
    select_beam_rough_topk_multilg<T, 8>
        <<<step_token_num, max_thread_per_block, 0, stream>>>(
            logits, logit_bias, seq_probs, seq_score, alive_seq, vocab_mask,
            src_token_id, can_idx, can_score, num_beam_can, vocab_size,
            max_step, length_norm, cur_step, diverse_lambda, end_id,
            src_seq_len);
  if (beam_size == 16)
    select_beam_rough_topk_multilg<T, 16>
        <<<step_token_num, max_thread_per_block, 0, stream>>>(
            logits, logit_bias, seq_probs, seq_score, alive_seq, vocab_mask,
            src_token_id, can_idx, can_score, num_beam_can, vocab_size,
            max_step, length_norm, cur_step, diverse_lambda, end_id,
            src_seq_len);
  if (beam_size == 32)
    select_beam_rough_topk_multilg<T, 32>
        <<<step_token_num, max_thread_per_block, 0, stream>>>(
            logits, logit_bias, seq_probs, seq_score, alive_seq, vocab_mask,
            src_token_id, can_idx, can_score, num_beam_can, vocab_size,
            max_step, length_norm, cur_step, diverse_lambda, end_id,
            src_seq_len);
}

template void select_beam_rough_topk_multilg_launcher<float>(
    const float* logits, const float* logit_bias, const float* seq_probs,
    const float* seq_score, const int* alive_seq, const int* vocab_mask,
    const int* src_token_id, int* can_idx, float* can_score, int* num_beam_can,
    int vocab_size, int max_step, float length_norm, int cur_step,
    int step_token_num, int max_thread_per_block, cudaStream_t stream,
    int beam_size, float diverse_lambda, int end_id, int src_seq_len);

template void select_beam_rough_topk_multilg_launcher<__half>(
    const __half* logits, const __half* logit_bias, const float* seq_probs,
    const float* seq_score, const int* alive_seq, const int* vocab_mask,
    const int* src_token_id, int* can_idx, float* can_score, int* num_beam_can,
    int vocab_size, int max_step, float length_norm, int cur_step,
    int step_token_num, int max_thread_per_block, cudaStream_t stream,
    int beam_size, float diverse_lambda, int end_id, int src_seq_len);

}  // namespace cuda
}  // namespace lightseq
