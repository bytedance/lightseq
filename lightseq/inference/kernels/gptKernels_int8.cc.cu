#include <random>

#include "common.h"
#include "gptKernels_int8.h"
#include "transformerKernels.h"
/**
@file
Implemented the cuda kernel function and its launcher
that required by GPT model.
Currently, fp16 and fp32 versions are provided
*/
namespace lightseq {
namespace cuda {
__forceinline__ __device__ int8_t float2int8(float x, float quant_scale) {
  float i8_f = x * quant_scale;
  int32_t i8 = floorf(i8_f + 0.5);
  i8 = i8 < -127 ? -127 : (i8 > 127 ? 127 : i8);
  return int8_t(i8);
}

template <typename T>
__global__ void ker_gpt_embedding_int8(const int8_t* token_emb,
                                       const T* pos_emb, const int* token_id,
                                       T* output, int* real_seq_len,
                                       int padding_id, int pos_offset,
                                       float dequant_scale) {
  int target_pos = blockIdx.x * gridDim.y + blockIdx.y;
  int tid = token_id[target_pos];
  if (tid == padding_id) {
    // for padding id
    output[target_pos * blockDim.x + threadIdx.x] = 0.f;
    return;
  }
  if (threadIdx.x == 0) {
    atomicAdd(real_seq_len + blockIdx.x, 1);
  }
  output[target_pos * blockDim.x + threadIdx.x] =
      T(token_emb[tid * blockDim.x + threadIdx.x]) * dequant_scale +
      pos_emb[(blockIdx.y + pos_offset) * blockDim.x + threadIdx.x];
}

/* fp16 version */
template <>
__global__ void ker_gpt_embedding_int8<__half>(
    const int8_t* token_emb, const __half* pos_emb, const int* token_id,
    __half* output, int* real_seq_len, int padding_id, int pos_offset,
    float dequant_scale) {
  int target_pos = blockIdx.x * gridDim.y + blockIdx.y;
  int tid = token_id[target_pos];
  half2* output_h = (half2*)output;

  if (tid == padding_id) {
    // for padding id
    output_h[target_pos * blockDim.x + threadIdx.x] = __float2half2_rn(0.f);
    return;
  }
  if (threadIdx.x == 0) {
    atomicAdd(real_seq_len + blockIdx.x, 1);
  }

  float2 te;
  char2 cte = ((const char2*)token_emb)[tid * blockDim.x + threadIdx.x];
  float2 pe = __half22float2(
      ((const half2*)
           pos_emb)[(blockIdx.y + pos_offset) * blockDim.x + threadIdx.x]);
  te.x = float(cte.x) * dequant_scale + pe.x;
  te.y = float(cte.y) * dequant_scale + pe.y;
  output_h[target_pos * blockDim.x + threadIdx.x] = __float22half2_rn(te);
}

template <typename T>
void ker_gpt_embedding_i8I_launcher(int batch_size, int batch_seq_len,
                                    int hidden_size, cudaStream_t stream,
                                    const int8_t* token_emb, const T* pos_emb,
                                    const int* token_id, T* output,
                                    int* real_seq_len, int padding_id,
                                    int pos_offset, float dequant_scale) {
  ker_gpt_embedding_int8<T>
      <<<dim3(batch_size, batch_seq_len), hidden_size, 0, stream>>>(
          token_emb, pos_emb, token_id, output, real_seq_len, padding_id,
          pos_offset, dequant_scale);
}

template <>
void ker_gpt_embedding_i8I_launcher<__half>(
    int batch_size, int batch_seq_len, int hidden_size, cudaStream_t stream,
    const int8_t* token_emb, const __half* pos_emb, const int* token_id,
    __half* output, int* real_seq_len, int padding_id, int pos_offset,
    float dequant_scale) {
  ker_gpt_embedding_int8<__half>
      <<<dim3(batch_size, batch_seq_len), hidden_size / 2, 0, stream>>>(
          token_emb, pos_emb, token_id, output, real_seq_len, padding_id,
          pos_offset, dequant_scale);
}

template void ker_gpt_embedding_i8I_launcher<float>(
    int batch_size, int batch_seq_len, int hidden_size, cudaStream_t stream,
    const int8_t* token_emb, const float* pos_emb, const int* token_id,
    float* output, int* real_seq_len, int padding_id, int pos_offset,
    float dequant_scale);

template void ker_gpt_embedding_i8I_launcher<__half>(
    int batch_size, int batch_seq_len, int hidden_size, cudaStream_t stream,
    const int8_t* token_emb, const __half* pos_emb, const int* token_id,
    __half* output, int* real_seq_len, int padding_id, int pos_offset,
    float dequant_scale);

__global__ void ker_ppl_i8I(const int8_t* logits, const int* input_ids,
                            const int* real_seq_len, float* ppl, int vocab_size,
                            float dequant_scale, bool in_col32) {
  int seq_len = real_seq_len[blockIdx.x];  // remove "eos"
  if (blockIdx.y >= seq_len - 1) {
    // will not contribute to ppl
    return;
  }

  int token_idx_in_batch = blockIdx.x * gridDim.y + blockIdx.y;
  int left_logit_idx = token_idx_in_batch * vocab_size + threadIdx.x;
  int right_logit_idx = (token_idx_in_batch + 1) * vocab_size;
  /*
  step 1. find max logit over the whole vocab
  */
  float max_logit = CUDA_FLOAT_INF_NEG;
  for (int idx = left_logit_idx; idx < right_logit_idx; idx += blockDim.x) {
    int logits_idx;
    if (in_col32) {
      int row_id = token_idx_in_batch;
      int col_id = idx - token_idx_in_batch * vocab_size;
      logits_idx = row_major2flat_col32(row_id, col_id, gridDim.x * gridDim.y,
                                        vocab_size);
    } else {
      logits_idx = idx;
    }
    max_logit = fmaxf(max_logit, (float)logits[logits_idx] * dequant_scale);
  }
  max_logit = blockReduceMax(max_logit);
  __shared__ float s_max_logit;
  if (threadIdx.x == 0) {
    s_max_logit = max_logit;
  }
  __syncthreads();

  /*
  step 2. compute the log probability for the given token,
  add it to the sequence's ppl
  */
  float sum_exp_logit = 0.f;
  for (int idx = left_logit_idx; idx < right_logit_idx; idx += blockDim.x) {
    int logits_idx;
    if (in_col32) {
      int row_id = token_idx_in_batch;
      int col_id = idx - token_idx_in_batch * vocab_size;
      logits_idx = row_major2flat_col32(row_id, col_id, gridDim.x * gridDim.y,
                                        vocab_size);
    } else {
      logits_idx = idx;
    }
    float lgt = fmaxf((float)logits[logits_idx] * dequant_scale - s_max_logit,
                      logit_thresh_min);
    sum_exp_logit += expf(lgt);
  }
  sum_exp_logit = blockReduceSum(sum_exp_logit);

  if (threadIdx.x == 0) {
    int token_id = input_ids[token_idx_in_batch + 1];
    int logits_idx;
    if (in_col32) {
      int row_id = token_idx_in_batch;
      int col_id = token_id;
      logits_idx = row_major2flat_col32(row_id, col_id, gridDim.x * gridDim.y,
                                        vocab_size);
    } else {
      logits_idx = token_idx_in_batch * vocab_size + token_id;
    }
    float log_prob = ((float)logits[logits_idx] * dequant_scale - s_max_logit -
                      logf(sum_exp_logit)) /
                     (float)(seq_len - 1);
    atomicAdd(ppl + blockIdx.x, -log_prob);
  }
}

void ker_ppl_i8I_launcher(int batch_size, int batch_seq_len,
                          int max_thread_per_block, cudaStream_t stream,
                          const int8_t* logits, const int* input_ids,
                          const int* real_seq_len, float* ppl, int vocab_size,
                          float dequant_scale, bool in_col32) {
  ker_ppl_i8I<<<dim3(batch_size, batch_seq_len), max_thread_per_block, 0,
                stream>>>(logits, input_ids, real_seq_len, ppl, vocab_size,
                          dequant_scale, in_col32);
}

template <typename T>
__global__ void ker_correlation_softmax_gpt_i32I(
    int32_t* correlation, T* output, const int* real_seq_len,
    const int batch_seq_len, float attn_scale, float dequant_scale) {
  int query_token_pos = blockIdx.y % batch_seq_len;
  if (query_token_pos >= real_seq_len[blockIdx.x]) {
    return;
  }

  int mask = 0;  // can see the token when mask=0
  if (threadIdx.x > query_token_pos || threadIdx.x >= batch_seq_len) {
    mask = 1;  // Can only see the token on the left side of it
  }

  int idx = (blockIdx.x * gridDim.y + blockIdx.y) * batch_seq_len + threadIdx.x;
  float val = threadIdx.x < batch_seq_len
                  ? ((float)correlation[idx] * attn_scale * dequant_scale *
                     dequant_scale)
                  : CUDA_FLOAT_INF_NEG;
  float max_val = blockReduceMax<float>(mask ? CUDA_FLOAT_INF_NEG : val);
  __shared__ float smax;
  if (threadIdx.x == 0) smax = max_val;
  __syncthreads();

  val = mask ? 0.f : expf(val - smax);
  float rsum = blockReduceSum<float>(val);
  __shared__ float ssum;
  if (threadIdx.x == 0) ssum = rsum;
  __syncthreads();

  if (threadIdx.x < batch_seq_len) output[idx] = (T)(val / ssum);
}

template <typename T>
void ker_correlation_softmax_gpt_i32I_launcher(
    int batch_size, int batch_seq_len, int head_num, cudaStream_t stream,
    int32_t* correlation, T* output, const int* real_seq_len, float attn_scale,
    float dequant_scale) {
  int block_dim = batch_seq_len;
  if (batch_seq_len < 1024) {
    block_dim = (batch_seq_len + 31) >> 5;
    block_dim *= 32;
  }

  ker_correlation_softmax_gpt_i32I<T>
      <<<dim3(batch_size, head_num * batch_seq_len), block_dim, 0, stream>>>(
          correlation, output, real_seq_len, batch_seq_len, attn_scale,
          dequant_scale);
}

template void ker_correlation_softmax_gpt_i32I_launcher<float>(
    int batch_size, int batch_seq_len, int head_num, cudaStream_t stream,
    int32_t* correlation, float* output, const int* real_seq_len,
    float attn_scale, float dequant_scale);

template void ker_correlation_softmax_gpt_i32I_launcher<__half>(
    int batch_size, int batch_seq_len, int head_num, cudaStream_t stream,
    int32_t* correlation, __half* output, const int* real_seq_len,
    float attn_scale, float dequant_scale);

template <int k>
__global__ void ker_topk_sample_i8I(const int8_t* logits, int* old_input_ids,
                                    int* new_input_ids, const int* real_seq_len,
                                    const int vocab_size,
                                    const int batch_seq_len, int logits_seq_len,
                                    int* unfinished, curandState* curandstate,
                                    int eos_id, float dequant_scale,
                                    bool in_col32) {
  int last_token_idx_in_batch = blockIdx.x * batch_seq_len + batch_seq_len - 1;

  /* add EOS to end if last token is EOS */
  if (old_input_ids[last_token_idx_in_batch] == eos_id) {
    int left_token_idx = blockIdx.x * batch_seq_len + threadIdx.x;
    int right_token_idx = (blockIdx.x + 1) * batch_seq_len;
    for (int idx = left_token_idx; idx < right_token_idx; idx += blockDim.x) {
      int new_idx = idx + blockIdx.x;
      new_input_ids[new_idx] = old_input_ids[idx];
    }
    if (threadIdx.x == 0) {
      // blockIdx.x * (batch_seq_len+1) + batch_seq_len
      new_input_ids[(blockIdx.x + 1) * (batch_seq_len + 1) - 1] = eos_id;
      old_input_ids[gridDim.x * batch_seq_len + blockIdx.x] = eos_id;
    }
    return;
  }
  int logits_token_idx_in_batch =
      blockIdx.x * logits_seq_len + logits_seq_len - 1;
  int left_logit_idx = logits_token_idx_in_batch * vocab_size + threadIdx.x;
  int right_logit_idx = (logits_token_idx_in_batch + 1) * vocab_size;

  /*
  step1. find max logit and rough Kth logit over the whole vocab
  */
  __shared__ float s_max_logit, s_topk_logit;
  float rough_top_kth_logit = CUDA_FLOAT_INF_NEG;
  for (int idx = left_logit_idx; idx < right_logit_idx; idx += blockDim.x) {
    int logits_idx;
    if (in_col32) {
      int row_id = logits_token_idx_in_batch;
      int col_id = idx - logits_token_idx_in_batch * vocab_size;
      logits_idx = row_major2flat_col32(row_id, col_id,
                                        gridDim.x * logits_seq_len, vocab_size);
    } else {
      logits_idx = idx;
    }
    rough_top_kth_logit =
        fmaxf(rough_top_kth_logit, (float)logits[logits_idx] * dequant_scale);
  }
  float max_logit = blockReduceMax(rough_top_kth_logit);
  rough_top_kth_logit = blockRoughTopK<float, k>(rough_top_kth_logit);
  if (threadIdx.x == 0) {
    s_topk_logit = rough_top_kth_logit;
    s_max_logit = max_logit;
  }
  __syncthreads();

  __shared__ int s_tid;

  if (k != 1) {
    /* step2 hold one logit per thread which larger than Kth logit and sample
     * from them */
    float topk_exp_sum, topk_exp = CUDA_FLOAT_INF_NEG;
    int topk_tid = vocab_size;
    int test_num = 0;
    __shared__ float s_topk_exp_sum;
    for (int idx = left_logit_idx; idx < right_logit_idx; idx += blockDim.x) {
      int logits_idx;
      if (in_col32) {
        int row_id = logits_token_idx_in_batch;
        int col_id = idx - logits_token_idx_in_batch * vocab_size;
        logits_idx = row_major2flat_col32(
            row_id, col_id, gridDim.x * logits_seq_len, vocab_size);
      } else {
        logits_idx = idx;
      }
      float logit = (float)logits[logits_idx] * dequant_scale;
      float logit_exp = expf(fmaxf(logit - s_max_logit, logit_thresh_min));
      if (logit >= s_topk_logit) test_num++;
      if (logit >= s_topk_logit && logit_exp > topk_exp) {
        topk_exp = logit_exp;
        topk_tid = idx - left_logit_idx + threadIdx.x;
      }
    }

    test_num = blockReduceSum(test_num);

    if (topk_tid == vocab_size) topk_exp = 0;
    topk_exp_sum = blockReduceSum(topk_exp);
    if (threadIdx.x == 0) {
      s_topk_exp_sum = topk_exp_sum;
    }
    __syncthreads();

    /* calculate cumulative probability */
    float topk_prob = topk_exp / s_topk_exp_sum;
    float prefix_sum_prob;
    typedef cub::BlockScan<float, 1024> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    BlockScan(temp_storage).InclusiveSum(topk_prob, prefix_sum_prob);

    __shared__ float random_x;
    if (threadIdx.x == 0) {
      random_x = curand_uniform(curandstate + blockIdx.x);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      s_tid = vocab_size;
    }
    __syncthreads();

    int threadID = threadIdx.x;
    __shared__ int s_threadID;
    __shared__ float s_max_prob;
    if (random_x > prefix_sum_prob) threadID = blockDim.x;
    threadID = blockReduceMin(threadID);
    float max_prob = blockReduceMax(topk_prob);
    if (threadIdx.x == 0) {
      s_threadID = threadID;
      s_max_prob = max_prob;
    }
    __syncthreads();
    if (threadIdx.x == s_threadID) {
      s_tid = topk_tid;
    }
    __syncthreads();

    if (s_tid == vocab_size && topk_prob == s_max_prob) {
      s_tid = topk_tid;
    }
    __syncthreads();
  } else {
    s_tid = vocab_size;
    for (int idx = left_logit_idx; idx < right_logit_idx; idx += blockDim.x) {
      int logits_idx;
      if (in_col32) {
        int row_id = logits_token_idx_in_batch;
        int col_id = idx - logits_token_idx_in_batch * vocab_size;
        logits_idx = row_major2flat_col32(
            row_id, col_id, gridDim.x * logits_seq_len, vocab_size);
      } else {
        logits_idx = idx;
      }
      float logit = (float)logits[logits_idx] * dequant_scale;
      if (logit == s_max_logit) {
        s_tid = idx - left_logit_idx + threadIdx.x;
      }
    }
    __syncthreads();
  }

  /* if new sampled tid is not EOS, set unfinish TRUE */
  if (threadIdx.x == 0) {
    if (s_tid != eos_id) unfinished[0] = 1;
  }

  /* step3 copy old_input_ids to new_input_ids and add new sampled ids */
  int left_token_idx = blockIdx.x * batch_seq_len + threadIdx.x;
  int right_token_idx = (blockIdx.x + 1) * batch_seq_len;
  for (int idx = left_token_idx; idx < right_token_idx; idx += blockDim.x) {
    int new_idx = idx + blockIdx.x;
    new_input_ids[new_idx] = old_input_ids[idx];
  }
  if (threadIdx.x == 0) {
    new_input_ids[(blockIdx.x + 1) * (batch_seq_len + 1) - 1] = s_tid;
    //  save the newly sampled ids to old_input_ids for next step inputs
    old_input_ids[gridDim.x * batch_seq_len + blockIdx.x] = s_tid;
  }
}

void ker_topk_sample_i8I_launcher(int batch_size, int batch_seq_len,
                                  int logits_seq_len, int max_thread_per_block,
                                  cudaStream_t stream, const int8_t* logits,
                                  int* old_input_ids, int* new_input_ids,
                                  const int* real_seq_len, const int vocab_size,
                                  const int k, int* unfinished,
                                  curandState* curandstate, int eos_id,
                                  float dequant_scale, bool in_col32) {
  if (k == 1)
    ker_topk_sample_i8I<1><<<batch_size, max_thread_per_block, 0, stream>>>(
        logits, old_input_ids, new_input_ids, real_seq_len, vocab_size,
        batch_seq_len, logits_seq_len, unfinished, curandstate, eos_id,
        dequant_scale, in_col32);
  else if (k == 2)
    ker_topk_sample_i8I<2><<<batch_size, max_thread_per_block, 0, stream>>>(
        logits, old_input_ids, new_input_ids, real_seq_len, vocab_size,
        batch_seq_len, logits_seq_len, unfinished, curandstate, eos_id,
        dequant_scale, in_col32);
  else if (k == 4)
    ker_topk_sample_i8I<4><<<batch_size, max_thread_per_block, 0, stream>>>(
        logits, old_input_ids, new_input_ids, real_seq_len, vocab_size,
        batch_seq_len, logits_seq_len, unfinished, curandstate, eos_id,
        dequant_scale, in_col32);
  else if (k == 8)
    ker_topk_sample_i8I<8><<<batch_size, max_thread_per_block, 0, stream>>>(
        logits, old_input_ids, new_input_ids, real_seq_len, vocab_size,
        batch_seq_len, logits_seq_len, unfinished, curandstate, eos_id,
        dequant_scale, in_col32);
  else if (k == 16)
    ker_topk_sample_i8I<16><<<batch_size, max_thread_per_block, 0, stream>>>(
        logits, old_input_ids, new_input_ids, real_seq_len, vocab_size,
        batch_seq_len, logits_seq_len, unfinished, curandstate, eos_id,
        dequant_scale, in_col32);
  else if (k == 32)
    ker_topk_sample_i8I<32><<<batch_size, max_thread_per_block, 0, stream>>>(
        logits, old_input_ids, new_input_ids, real_seq_len, vocab_size,
        batch_seq_len, logits_seq_len, unfinished, curandstate, eos_id,
        dequant_scale, in_col32);
  else {
    throw std::invalid_argument("topk argument should be in [1,2,4,8,16,32]");
  }
}

__global__ void ker_topp_sample_i8I(const int8_t* logits, int* old_input_ids,
                                    int* new_input_ids, const int* real_seq_len,
                                    const int vocab_size,
                                    const int batch_seq_len, int logits_seq_len,
                                    int* unfinished, float p,
                                    curandState* curandstate, int eos_id,
                                    float dequant_scale, bool in_col32) {
  int token_idx_in_batch = blockIdx.x * batch_seq_len + batch_seq_len - 1;

  /* add EOS to end if last token is EOS */
  if (old_input_ids[token_idx_in_batch] == eos_id) {
    int left_token_idx = blockIdx.x * batch_seq_len + threadIdx.x;
    int right_token_idx = (blockIdx.x + 1) * batch_seq_len;
    for (int idx = left_token_idx; idx < right_token_idx; idx += blockDim.x) {
      int new_idx = idx + blockIdx.x;
      new_input_ids[new_idx] = old_input_ids[idx];
    }
    if (threadIdx.x == 0) {
      new_input_ids[(blockIdx.x + 1) * (batch_seq_len + 1) - 1] = eos_id;
      old_input_ids[gridDim.x * batch_seq_len + blockIdx.x] = eos_id;
    }
    return;
  }
  int logits_token_idx_in_batch =
      blockIdx.x * logits_seq_len + logits_seq_len - 1;
  int left_logit_idx = logits_token_idx_in_batch * vocab_size + threadIdx.x;
  int right_logit_idx = (logits_token_idx_in_batch + 1) * vocab_size;

  /*
  step1. find max logit in each thread and sample from these probs with nucleus
  sampling
  */
  __shared__ float s_max_logit;
  float max_logit = CUDA_FLOAT_INF_NEG;
  for (int idx = left_logit_idx; idx < right_logit_idx; idx += blockDim.x) {
    int logits_idx;
    if (in_col32) {
      int row_id = logits_token_idx_in_batch;
      int col_id = idx - logits_token_idx_in_batch * vocab_size;
      logits_idx = row_major2flat_col32(row_id, col_id,
                                        gridDim.x * logits_seq_len, vocab_size);
    } else {
      logits_idx = idx;
    }
    max_logit = fmaxf(max_logit, (float)logits[logits_idx] * dequant_scale);
  }
  float max_logit_array[1];
  max_logit_array[0] = max_logit;
  typedef cub::BlockRadixSort<float, 1024, 1> BlockRadixSort;
  __shared__ typename BlockRadixSort::TempStorage sort_temp_storage;
  BlockRadixSort(sort_temp_storage).SortDescending(max_logit_array);
  float presum_max_logit_exp;
  max_logit = max_logit_array[0];

  float block_max_logit = blockReduceMax(max_logit);
  if (threadIdx.x == 0) {
    s_max_logit = block_max_logit;
  }
  __syncthreads();

  float biased_logit_exp =
      expf(fmaxf(max_logit - s_max_logit, logit_thresh_min));

  typedef cub::BlockScan<float, 1024> BlockScan;
  __shared__ typename BlockScan::TempStorage presum_temp_storage;
  BlockScan(presum_temp_storage)
      .InclusiveSum(biased_logit_exp, presum_max_logit_exp);

  float topp_exp_threshold;
  if (threadIdx.x == blockDim.x - 1) {
    topp_exp_threshold = p * presum_max_logit_exp;
  }
  __shared__ float s_presum_logit_exp_threshold;
  if (presum_max_logit_exp > topp_exp_threshold) {
    presum_max_logit_exp = CUDA_FLOAT_INF_NEG;
  }
  float logit_exp_threshold = blockReduceMax(presum_max_logit_exp);
  if (threadIdx.x == 0) {
    s_presum_logit_exp_threshold = logit_exp_threshold;
  }
  __syncthreads();

  __shared__ float s_logit_threshold;
  if (presum_max_logit_exp == s_presum_logit_exp_threshold) {
    s_logit_threshold = max_logit;
  }
  __syncthreads();

  /* step2 hold one logit per thread and sample
   * from them */
  float topk_exp_sum, topk_exp = CUDA_FLOAT_INF_NEG;
  int topk_tid = vocab_size;
  int test_num = 0;
  __shared__ float s_topk_exp_sum;
  for (int idx = left_logit_idx; idx < right_logit_idx; idx += blockDim.x) {
    int logits_idx;
    if (in_col32) {
      int row_id = logits_token_idx_in_batch;
      int col_id = idx - logits_token_idx_in_batch * vocab_size;
      logits_idx = row_major2flat_col32(row_id, col_id,
                                        gridDim.x * logits_seq_len, vocab_size);
    } else {
      logits_idx = idx;
    }
    float logit = (float)logits[logits_idx] * dequant_scale;
    float logit_exp = expf(fmaxf(logit - s_max_logit, logit_thresh_min));
    if (logit >= s_logit_threshold) test_num++;
    if (logit >= s_logit_threshold && logit_exp > topk_exp) {
      topk_exp = logit_exp;
      topk_tid = idx - left_logit_idx + threadIdx.x;
    }
  }

  test_num = blockReduceSum(test_num);

  if (topk_tid == vocab_size) topk_exp = 0;
  topk_exp_sum = blockReduceSum(topk_exp);
  if (threadIdx.x == 0) {
    s_topk_exp_sum = topk_exp_sum;
  }
  __syncthreads();

  /* calculate cumulative probability */
  float topk_prob = topk_exp / s_topk_exp_sum;
  float prefix_sum_prob;
  BlockScan(presum_temp_storage).InclusiveSum(topk_prob, prefix_sum_prob);

  __shared__ float random_x;
  if (threadIdx.x == 0) {
    random_x = curand_uniform(curandstate + blockIdx.x);
  }
  __syncthreads();

  __shared__ int s_tid;
  if (threadIdx.x == 0) {
    s_tid = vocab_size;
  }
  __syncthreads();

  int threadID = threadIdx.x;
  __shared__ int s_threadID;
  __shared__ float s_max_prob;
  if (random_x > prefix_sum_prob) threadID = blockDim.x;
  threadID = blockReduceMin(threadID);
  float max_prob = blockReduceMax(topk_prob);
  if (threadIdx.x == 0) {
    s_threadID = threadID;
    s_max_prob = max_prob;
  }
  __syncthreads();
  if (threadIdx.x == s_threadID) {
    s_tid = topk_tid;
  }
  __syncthreads();

  if (s_tid == vocab_size && topk_prob == s_max_prob) {
    s_tid = topk_tid;
  }
  __syncthreads();

  /* if new sampled tid is not EOS, set unfinish TRUE */
  if (threadIdx.x == 0) {
    if (s_tid != eos_id) unfinished[0] = 1;
  }

  /* step3 copy old_input_ids to new_input_ids and add new sampled ids */
  int left_token_idx = blockIdx.x * batch_seq_len + threadIdx.x;
  int right_token_idx = (blockIdx.x + 1) * batch_seq_len;
  for (int idx = left_token_idx; idx < right_token_idx; idx += blockDim.x) {
    int new_idx = idx + blockIdx.x;
    new_input_ids[new_idx] = old_input_ids[idx];
  }
  if (threadIdx.x == 0) {
    new_input_ids[(blockIdx.x + 1) * (batch_seq_len + 1) - 1] = s_tid;
    //  save the newly sampled ids to old_input_ids for next step inputs
    old_input_ids[gridDim.x * batch_seq_len + blockIdx.x] = s_tid;
  }
}

void ker_topp_sample_i8I_launcher(int batch_size, int batch_seq_len,
                                  int logits_seq_len, int max_thread_per_block,
                                  cudaStream_t stream, const int8_t* logits,
                                  int* old_input_ids, int* new_input_ids,
                                  const int* real_seq_len, const int vocab_size,
                                  const float p, int* unfinished,
                                  curandState* curandstate, int eos_id,
                                  float dequant_scale, bool in_col32) {
  ker_topp_sample_i8I<<<batch_size, max_thread_per_block, 0, stream>>>(
      logits, old_input_ids, new_input_ids, real_seq_len, vocab_size,
      batch_seq_len, logits_seq_len, unfinished, p, curandstate, eos_id,
      dequant_scale, in_col32);
}

template <typename T>
__global__ void ker_arrange_qkv_with_cache_i8I_i8O(
    const int8_t* ori_qkv, const T* qkv_bias, int8_t* new_q, int8_t* new_k,
    int8_t* k_cache, int8_t* new_v, int8_t* v_cache, int batch_seq_len,
    int dim_per_head, int head_num, float dequant_scale, float quant_scale,
    bool in_col32) {
  int hidden_size = head_num * dim_per_head;
  int batch_size = gridDim.x / batch_seq_len;
  int batch_id = blockIdx.x / batch_seq_len;
  int token_id = blockIdx.x % batch_seq_len;
  int head_id = threadIdx.x / dim_per_head;
  int dim_id = threadIdx.x % dim_per_head;
  int target_id = targetid_4dim(batch_id, head_id, token_id, dim_id, head_num,
                                batch_seq_len, dim_per_head);
  int8_t new_val;

  if (token_id < batch_seq_len - 1) {
    int old_target_id =
        targetid_4dim(batch_id, head_id, token_id, dim_id, head_num,
                      batch_seq_len - 1, dim_per_head);
    if (blockIdx.y == 0) return;
    if (blockIdx.y == 1) new_val = k_cache[old_target_id];
    if (blockIdx.y == 2) new_val = v_cache[old_target_id];
  } else {
    int qkv_index;
    if (in_col32) {
      int row_id = batch_id;
      int col_id = blockIdx.y * hidden_size + threadIdx.x;
      qkv_index = row_major2flat_col32(row_id, col_id, batch_size,
                                       gridDim.y * hidden_size);
    } else {
      qkv_index =
          (batch_id * gridDim.y + blockIdx.y) * hidden_size + threadIdx.x;
    }
    float tmp_val = float(ori_qkv[qkv_index]) * dequant_scale +
                    __ldg(&qkv_bias[blockIdx.y * hidden_size + threadIdx.x]);
    new_val = float2int8(tmp_val, quant_scale);
    if (blockIdx.y == 0) {
      target_id = targetid_4dim(batch_id, head_id, 0, dim_id, head_num, 1,
                                dim_per_head);
    }
  }

  if (blockIdx.y == 0) new_q[target_id] = new_val;
  if (blockIdx.y == 1) new_k[target_id] = new_val;
  if (blockIdx.y == 2) {
    new_v[target_id] = new_val;
  }
}

template <>
__global__ void ker_arrange_qkv_with_cache_i8I_i8O<__half>(
    const int8_t* ori_qkv, const __half* qkv_bias, int8_t* new_q, int8_t* new_k,
    int8_t* k_cache, int8_t* new_v, int8_t* v_cache, int batch_seq_len,
    int dim_per_head, int head_num, float dequant_scale, float quant_scale,
    bool in_col32) {
  int hidden_size = head_num * dim_per_head;
  int batch_size = gridDim.x / batch_seq_len;
  int batch_id = blockIdx.x / batch_seq_len;
  int token_id = blockIdx.x % batch_seq_len;
  int head_id = threadIdx.x / dim_per_head;
  int dim_id = threadIdx.x % dim_per_head;
  int target_id = targetid_4dim(batch_id, head_id, token_id, dim_id, head_num,
                                batch_seq_len, dim_per_head);
  int2 new_val;
  int8_t* p_new_val = (int8_t*)(&new_val);
  const int2* p_ori_qkv = (const int2*)ori_qkv;
  const float4* p_bias = (const float4*)qkv_bias;
  const int2* p_k_cache = (const int2*)k_cache;
  const int2* p_v_cache = (const int2*)v_cache;
  int2* p_new_q = (int2*)new_q;
  int2* p_new_k = (int2*)new_k;
  int2* p_new_v = (int2*)new_v;

  if (token_id < batch_seq_len - 1) {
    int old_target_id =
        targetid_4dim(batch_id, head_id, token_id, dim_id, head_num,
                      batch_seq_len - 1, dim_per_head);
    if (blockIdx.y == 0) return;
    if (blockIdx.y == 1) new_val = p_k_cache[old_target_id];
    if (blockIdx.y == 2) new_val = p_v_cache[old_target_id];
  } else {
    int qkv_index;
    if (in_col32) {
      int row_id = batch_id;
      int col_id = (blockIdx.y * hidden_size + threadIdx.x) << 3;
      qkv_index = row_major2flat_col32(row_id, col_id, batch_size,
                                       (gridDim.y * hidden_size) << 3) >>
                  3;
    } else {
      qkv_index =
          (batch_id * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
    }
    int2 ori_qkv8 = p_ori_qkv[qkv_index];
    float4 bias8 = __ldg(&p_bias[blockIdx.y * blockDim.x + threadIdx.x]);
    int8_t* p_ori_qkv8 = (int8_t*)(&ori_qkv8);
    __half* p_bias8 = (__half*)(&bias8);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      p_new_val[i] =
          float2int8(float(p_ori_qkv8[i]) * dequant_scale + float(p_bias8[i]),
                     quant_scale);
    }
    if (blockIdx.y == 0) {
      target_id = targetid_4dim(batch_id, head_id, 0, dim_id, head_num, 1,
                                dim_per_head);
    }
  }

  if (blockIdx.y == 0) p_new_q[target_id] = new_val;
  if (blockIdx.y == 1) p_new_k[target_id] = new_val;
  if (blockIdx.y == 2) p_new_v[target_id] = new_val;
}

template <typename T>
void ker_arrange_qkv_with_cache_i8I_i8O_launcher(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const int8_t* ori_qkv, const T* qkv_bias, int8_t* new_q, int8_t* new_k,
    int8_t* k_cache, int8_t* new_v, int8_t* v_cache, int batch_seq_len,
    int dim_per_head, int head_num, float dequant_scale, float quant_scale,
    bool in_col32) {
  ker_arrange_qkv_with_cache_i8I_i8O<T>
      <<<dim3(batch_token_num, 3), hidden_size, 0, stream>>>(
          ori_qkv, qkv_bias, new_q, new_k, k_cache, new_v, v_cache,
          batch_seq_len, dim_per_head, head_num, dequant_scale, quant_scale,
          in_col32);
}

template <>
void ker_arrange_qkv_with_cache_i8I_i8O_launcher<__half>(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const int8_t* ori_qkv, const __half* qkv_bias, int8_t* new_q, int8_t* new_k,
    int8_t* k_cache, int8_t* new_v, int8_t* v_cache, int batch_seq_len,
    int dim_per_head, int head_num, float dequant_scale, float quant_scale,
    bool in_col32) {
  ker_arrange_qkv_with_cache_i8I_i8O<__half>
      <<<dim3(batch_token_num, 3), hidden_size / 8, 0, stream>>>(
          ori_qkv, qkv_bias, new_q, new_k, k_cache, new_v, v_cache,
          batch_seq_len, dim_per_head / 8, head_num, dequant_scale, quant_scale,
          in_col32);
}

template void ker_arrange_qkv_with_cache_i8I_i8O_launcher<float>(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const int8_t* ori_qkv, const float* qkv_bias, int8_t* new_q, int8_t* new_k,
    int8_t* k_cache, int8_t* new_v, int8_t* v_cache, int batch_seq_len,
    int dim_per_head, int head_num, float dequant_scale, float quant_scale,
    bool in_col32);

template void ker_arrange_qkv_with_cache_i8I_i8O_launcher<__half>(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const int8_t* ori_qkv, const __half* qkv_bias, int8_t* new_q, int8_t* new_k,
    int8_t* k_cache, int8_t* new_v, int8_t* v_cache, int batch_seq_len,
    int dim_per_head, int head_num, float dequant_scale, float quant_scale,
    bool in_col32);

template <typename T>
__global__ void ker_attention_mask_weights_i32I(
    int32_t* correlation, T* output, const int* real_seq_len, int dst_seq_len,
    int src_seq_len, float attn_scale, float dequant_scale) {
  int query_token_pos = blockIdx.y % dst_seq_len + src_seq_len - dst_seq_len;
  if (query_token_pos >= real_seq_len[blockIdx.x]) {
    return;
  }
  int mask = 0;  // can see the token when mask=0
  if (threadIdx.x > query_token_pos) {
    mask = 1;  // Can only see the token on the left side of it
  }

  int idx = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
  float val =
      (float)correlation[idx] * attn_scale * dequant_scale * dequant_scale;
  float max_val = blockReduceMax<float>(mask ? CUDA_FLOAT_INF_NEG : val);
  __shared__ float smax;
  if (threadIdx.x == 0) smax = max_val;
  __syncthreads();

  val = mask ? 0.f : expf(fmaxf(logit_thresh_min, val - smax));
  float rsum = blockReduceSum<float>(val);
  __shared__ float ssum;
  if (threadIdx.x == 0) ssum = rsum;
  __syncthreads();

  output[idx] = (T)(val / (ssum + epsilon));
}

template <typename T>
void ker_attention_mask_weights_i32I_launcher(
    int batch_size, int dst_seq_len, int src_seq_len, int head_num,
    cudaStream_t stream, int32_t* correlation, T* output,
    const int* real_seq_len, float attn_scale, float dequant_scale) {
  ker_attention_mask_weights_i32I<T>
      <<<dim3(batch_size, head_num * dst_seq_len), src_seq_len, 0, stream>>>(
          correlation, output, real_seq_len, dst_seq_len, src_seq_len,
          attn_scale, dequant_scale);
}

template void ker_attention_mask_weights_i32I_launcher<float>(
    int batch_size, int dst_seq_len, int src_seq_len, int head_num,
    cudaStream_t stream, int32_t* correlation, float* output,
    const int* real_seq_len, float attn_scale, float dequant_scale);

template void ker_attention_mask_weights_i32I_launcher<__half>(
    int batch_size, int dst_seq_len, int src_seq_len, int head_num,
    cudaStream_t stream, int32_t* correlation, __half* output,
    const int* real_seq_len, float attn_scale, float dequant_scale);

}  // namespace cuda
}  // namespace lightseq
