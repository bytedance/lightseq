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

}  // namespace cuda
}  // namespace lightseq
