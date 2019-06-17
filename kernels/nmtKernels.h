#pragma once

#include "src/custom/transformer/kernels/common.h"

namespace lab {
namespace nmt {

const float logit_thresh_max = 64.f;
const float logit_thresh_min = -64.f;
const float min_log_probability = -10000.f;
const float epsilon = 0.000001;

template <int beam_size>
__global__ void ker_update_new_seq_probs(float* logits, const float* logit_bias,
                                         const float* seq_probs, int vocab_size,
                                         int* can_idx, float* can_probs,
                                         int* ncan, float* finished_scores,
                                         float* cur_finished_scores,
                                         float length_norm) {
  /**
  @brief
  one block for one beam, compute the seq probability ended with every token in
  vocab, base on the previous seq probability and current step's logit, select
  rough topK candidate.

  @thread
  gridDim.x = batch_size * beam_size
  blockDim.x = max_thread_per_block

  @param
  logits: [batch_size, beam_size, vocab_size], cur step logit
  seq_probs: [batch_size, beam_size], prefix sequence log probability
  vocab_size: vocabulary size
  can_idx: [batch_size, beam_size, vocab_size], topk candidate's index
  can_probs: [batch_size, beam_size, vocab_size], topk candidate's log
  probability ncan: [1+batch_size*beam_size], number of topk candidate
  finished_scores: [batch_size], best finished score util now
  cur_finished_scores: [batch_size, beam_size], cur step's finished score
  */

  /* step1: compute -log(sum(exp(logits))) + seq_probs */
  const int block_start = blockIdx.x * vocab_size;
  const int left_idx = block_start + threadIdx.x;
  int right_idx = (blockIdx.x + 1) * vocab_size - 1;  // prevent END
  float local_max = logit_thresh_min;
  float local_sum = 0;
  for (int i = left_idx; i < right_idx; i += blockDim.x) {
    float lgt =
        max(min(logits[i] + logit_bias[i - block_start], logit_thresh_max),
            logit_thresh_min);
    local_max = max(local_max, lgt);
    local_sum += expf(lgt);
  }

  __shared__ float s_sum;
  float block_sum = blockReduceSum(local_sum);
  if (threadIdx.x == 0) {
    float eos_lgt = max(
        min(logits[right_idx] + logit_bias[vocab_size - 1], logit_thresh_max),
        logit_thresh_min);
    s_sum = seq_probs[blockIdx.x] - logf(block_sum + expf(eos_lgt));
  }

  /*
  step2: compute a rough top k-th value of logits, saved into smem, s_topk.
         notice, rough but safe
  */
  __shared__ float s_topk;  // rough top k-th value of logits
  __shared__ int s_ncan;    // candidate number in this block
  local_max = blockRoughTopK<float, beam_size>(local_max);
  if (threadIdx.x == 0) {
    // s_topk = min(logit_thresh_max, local_max);
    s_topk = local_max;
    s_ncan = 0;
  }
  //__syncthreads();

  /*
  step3 : select the candidate token with logits bigger than s_topk,
          compute the seq probability ended with them,
      save the probability, token_index, selected token number.
  */
  int idx = left_idx;
  int batch_id = blockIdx.x / beam_size;
  int batch_start_pos = batch_id * beam_size * vocab_size;
  right_idx -= 2;      // prevent <unk> <start> <end>
  __shared__ int l_n;  // current iteration candidate number
  for (int iter = 0; iter < (vocab_size + blockDim.x - 1) / blockDim.x;
       iter++) {
    // zero the counter
    if (threadIdx.x == 0) l_n = 0;
    __syncthreads();

    // get the value, evaluate the predicate, and
    // increment the counter if needed
    float lgt;
    int pos;

    if (idx < right_idx) {
      lgt = logits[idx] + logit_bias[idx - block_start];
      if (lgt >= s_topk)
        // pos: relative pos inside this iteration
        pos = atomicAdd(&l_n, 1);
    }
    __syncthreads();

    // leader increments the global counter
    if (threadIdx.x == 0) {
      atomicAdd(&s_ncan, l_n);
      l_n = atomicAdd(ncan, l_n);
    }
    __syncthreads();

    // threads with true predicates write their elements
    if (idx < right_idx && lgt >= s_topk) {
      pos += l_n;  // increment local pos by global counter
      can_probs[pos] =
          max(min(lgt, logit_thresh_max) + s_sum, min_log_probability + 1.f) +
          batch_id * min_log_probability;
      // can_probs[pos] = s_topk;
      can_idx[pos] = idx - batch_start_pos;
    }
    __syncthreads();

    idx += blockDim.x;
  }

  if (threadIdx.x == 0) {
    ncan[blockIdx.x + 1] = s_ncan;
    float lgt =
        logits[(blockIdx.x + 1) * vocab_size - 1];  // the EOS token logit
    float score = max(min(lgt, logit_thresh_max) + s_sum, min_log_probability) *
                  length_norm;
    cur_finished_scores[blockIdx.x] = score;
    atomicMaxFloat(finished_scores + batch_id, score);
  }
}
__global__ void kerBiasRelu(float* first_output, const float* first_bias,
                            unsigned int inner_size);

__global__ void kerBiasAdd(float* ffn_input, const float* second_out,
                           const float* second_bias);

__global__ void ker_norm_layer(float* matrix, const float* scale,
                               const float* bias);

__global__ void ker_norm_layer2(const float* input, float* output,
                                const float* scale, const float* bias);

__global__ void ker_norm_layer3(float* input, float* output, const float* scale,
                                const float* bias, const float* residual_bias);

__global__ void ker_enc_embedding(const float* token_emb, const float* pos_emb,
                                  const int* token_id, float* output,
                                  int* padding_mask, int padding_id);

__global__ void ker_dec_embedding(const float* token_emb, const float* pos_emb,
                                  const int* token_id, float* output, int step,
                                  int max_step, int vocab_size);

__global__ void ker_arrange_encself_qkv(const float* ori_qkv,
                                        const float* qkv_bias, float* new_qkv,
                                        int max_batch_dim, int batch_seq_len,
                                        int dim_per_head, int head_num);

__global__ void ker_arrange_decself_qkv(const float* ori_qkv,
                                        const float* qkv_bias, float* new_q,
                                        float* new_k, float* new_v,
                                        int head_num, int dim_per_head,
                                        int max_step, int step_id);

__global__ void ker_refresh_cache(const int* num_can_per_beam,
                                  const int* can_idx, float* self_k_bgeem,
                                  float* self_v_bgeem, float* new_self_k_bgeem,
                                  float* new_self_v_bgeem,
                                  int self_k_bgeem_offset, int beam_size,
                                  int dim_per_head, int head_num,
                                  int vocab_size, int cur_step, int max_step);

__global__ void ker_arrange_encdec_kv(const float* ori_kv, const float* kv_bias,
                                      float* new_k, float* new_v,
                                      int offset_per_layer, int batch_seq_len,
                                      int dim_per_head, int head_num);

__global__ void ker_arrange_encdec_q(const float* ori_q, const float* q_bias,
                                     float* new_q, int beam_size,
                                     int dim_per_head, int head_num);

__global__ void ker_correlation_softmax_encself(float* correlation,
                                                const int* src_padding_mask);

__global__ void ker_correlation_softmax_decself(float* correlation);

__global__ void ker_correlation_softmax_encdec(float* correlation,
                                               const int* src_padding_mask);

__global__ void ker_arrange_atten_output(const float* ori_q, float* new_q,
                                         int beam_size, int dim_per_head,
                                         int head_num);

__global__ void ker_refresh_result(
    const int* can_idx, const float* can_probs, const int* num_can_per_beam,
    const int* old_alive_seq, int* new_alive_seq, float* alive_seq_probs,
    const float* finished_scores, const float* cur_finished_scores,
    int* finished_seq, int* num_finish_beam, int vocab_size, int cur_step,
    float max_length_norm);

__global__ void ker_write_trg_tokenid(const int* input, int* output,
                                      int max_step);

}  // namespace nmt
}  // namespace lab
