#pragma once

#include "src/custom/transformer/kernels/common.h"

namespace lab {
namespace nmt {

const float logit_thresh_max = 64.f;
const float logit_thresh_min = -64.f;
const float min_log_probability = -10000.f;
const float epsilon = 0.000001;

template <int beam_size>
__global__ void select_beam_rough_topk(float* logits, const float* logit_bias,
                                         const float* seq_probs, 
					 const float* seq_score,
					 const int* alive_seq,
                                         int* can_idx, float* can_score,
                                         int* num_beam_can, int vocab_size, 
					 int max_step, float length_norm, int cur_step) {
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
  logit_bias: [vocab_size], logit bias
  seq_probs: [batch_size, beam_size], prefix sequence sum log probability
  seq_score: [batch_size, beam_size], prefix sequence score
  alive_seq: [batch_size, beam_size, max_step], prefix sequence id
  can_idx: [batch_size, beam_size, vocab_size], topk candidate's index
  can_score: [batch_size, beam_size, vocab_size], topk candidate's score
  num_beam_can: [1 + batch_size * beam_size]. 
      the first ele save the number of topk candidate of the whole batch
      the remaining batch_size * beam_size ele save the number of topk candidate of each beam
  */

  if(alive_seq[blockIdx.x * max_step + cur_step] == vocab_size -1) {
    // this is a finished beam
    if (threadIdx.x == 0) {
      num_beam_can[blockIdx.x + 1] = 1; // generate one candidate
      int pos = atomicAdd(num_beam_can, 1); // get a candidate pos
      can_score[pos] = seq_score[blockIdx.x]; // this beam's score will not be change
      can_idx[pos] = vocab_size - 1 + (blockIdx.x % beam_size) * vocab_size; // EOS
    }
    return;
  }

  /* step1: compute each thread's max_logit and sum_exp_logit, store in rough_top_kth_logit, sum_exp_logit */
  const int block_start = blockIdx.x * vocab_size;
  const int left_idx = block_start + threadIdx.x;
  const int right_idx = (blockIdx.x + 1) * vocab_size;
  float rough_top_kth_logit = CUDA_FLOAT_INF_NEG;
  float sum_exp_logit = 0;
  for (int i = left_idx; i < right_idx; i += blockDim.x) {
    float lgt = logits[i] + logit_bias[i - block_start];
    rough_top_kth_logit = fmaxf(rough_top_kth_logit, lgt);
  }  
  float max_logit = blockReduceMax(rough_top_kth_logit);  
  __shared__ float s_max_logit;
  if (threadIdx.x == 0) {
    s_max_logit = max_logit;
  }
  __syncthreads();
  for (int i = left_idx; i < right_idx; i += blockDim.x) {
    float lgt = fmaxf(logits[i] + logit_bias[i - block_start] - s_max_logit, logit_thresh_min);
    sum_exp_logit += expf(lgt);
  }

  /*
  step2: compute rough top-kth-logits and sum_exp_logit among the whole beam, saved into s_topk and
      s_log_prob_base
  */
  __shared__ float s_log_prob_base;  // prefix sequence log prob - log_sum_exp_logit
  __shared__ float s_topk;  // rough top k-th value of logits
  __shared__ int num_cur_beam_can;    // candidate number for this beam
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
  int unk_vocab_id = vocab_size - 3; // last three element: unk, start, eos
  __shared__ int l_n;  // current iteration candidate number
  for (int iter = 0; iter < (vocab_size + blockDim.x - 1) / blockDim.x;
       iter++) {
    // zero the counter
    if (threadIdx.x == 0) l_n = 0;
    __syncthreads();

    float lgt = logit_thresh_min - 1.f; // min s_topk is logit_thresh_min
    int pos;
    int vocab_id = idx - block_start;

    if ((vocab_id < vocab_size) && (vocab_id != unk_vocab_id)) {
      lgt = logits[idx] + logit_bias[vocab_id];
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
      can_score[pos] =
          fmaxf((lgt + s_log_prob_base) * length_norm, min_log_probability + 1.f) +
          batch_id * min_log_probability;
      can_idx[pos] = idx - batch_start_pos;
    }
    __syncthreads();
    idx += blockDim.x;
  }
  if (threadIdx.x == 0) {
    num_beam_can[blockIdx.x + 1] = num_cur_beam_can;
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
    const int* can_idx, const float* can_score, const int* num_can_per_beam,
    const int* old_alive_seq, int* new_alive_seq, float* seq_probs,
    float* seq_score, int* num_finish_beam, int vocab_size, int cur_step,
    float length_norm);

__global__ void ker_write_trg_tokenid_pos_penalty(const int* alive_seq,
		int* output, int max_step, int beam_size);

__global__ void ker_write_trg_tokenid_neg_penalty(const int* alive_seq, const float* seq_score,
		int* output, int max_step, int beam_size, int vocab_size);

__global__ void ker_write_topk_result(const int* alive_seq, 
        float* seq_score, int* res_seq, 
	int vocab_size, int max_step, int beam_size);

}  // namespace nmt
}  // namespace lab
