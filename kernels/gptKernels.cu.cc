#include "src/custom/transformer/kernels/common.h"
#include "src/custom/transformer/kernels/gptKernels.h"
#include "src/custom/transformer/kernels/nmtKernels.h"

namespace lab {
namespace nmt {

template <typename T>
__global__ void ker_gpt_embedding(const T *token_emb, const T *pos_emb,
                                  const int *token_id, T *output,
                                  int *real_seq_len, int padding_id) {
  /**
  @brief
  for encoder, look up token embedding, add position embedding

  @thread
  gridDim.x = batch_size
  gridDim.y = batch_seq_len
  blockDim.x = hidden_size

  @param
  token_emb: [vocab_size, hidden_size]
  pos_emb: [max_step, hidden_size]
  token_id: input token id, [batch_size, batch_seq_len]
  output: result, [batch_size, batch_seq_len, hidden_size]
  real_seq_len: record seq len exclude padding, [batch_size]
  padding_id, the padding_id, default 0
  */
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
      token_emb[tid * blockDim.x + threadIdx.x] +
      pos_emb[blockIdx.y * blockDim.x + threadIdx.x];
}

template <>
__global__ void ker_gpt_embedding<__half>(const __half *token_emb, const __half *pos_emb,
                                  const int *token_id, __half *output,
                                  int *real_seq_len, int padding_id) {
  int target_pos = blockIdx.x * gridDim.y + blockIdx.y;
  int tid = token_id[target_pos];  
  half2* output_h = (half2 *) output;

  if (tid == padding_id) {
    // for padding id
    output_h[target_pos * blockDim.x + threadIdx.x] = __float2half2_rn(0.f);
    return;
  }
  if (threadIdx.x == 0) {
    atomicAdd(real_seq_len + blockIdx.x, 1);
  }
  
  float2 te = __half22float2(
    ((const half2 *)token_emb)[tid * blockDim.x + threadIdx.x]);
  float2 pe = __half22float2(
    ((const half2 *)pos_emb)[blockIdx.y * blockDim.x + threadIdx.x]);
  te.x += pe.x;
  te.y += pe.y;
  output_h[target_pos * blockDim.x + threadIdx.x] =
    __float22half2_rn(te);
}

template <typename T>
void ker_gpt_embedding_launcher(int batch_size, int batch_seq_len,
                                int hidden_size, cudaStream_t stream,
                                const T *token_emb, const T *pos_emb,
                                const int *token_id, T *output,
                                int *real_seq_len, int padding_id) {
  ker_gpt_embedding<
      T><<<dim3(batch_size, batch_seq_len), hidden_size, 0, stream>>>(
      token_emb, pos_emb, token_id, output, real_seq_len, padding_id);
}

template <>
void ker_gpt_embedding_launcher<__half>(int batch_size, int batch_seq_len,
                                int hidden_size, cudaStream_t stream,
                                const __half *token_emb, const __half *pos_emb,
                                const int *token_id, __half *output,
                                int *real_seq_len, int padding_id) {
  ker_gpt_embedding<
      __half><<<dim3(batch_size, batch_seq_len), hidden_size / 2, 0, stream>>>(
      token_emb, pos_emb, token_id, output, real_seq_len, padding_id);
}

template void ker_gpt_embedding_launcher<float>(
    int batch_size, int batch_seq_len, int hidden_size, cudaStream_t stream,
    const float *token_emb, const float *pos_emb, const int *token_id,
    float *output, int *real_seq_len, int padding_id);

template void ker_gpt_embedding_launcher<__half>(
    int batch_size, int batch_seq_len, int hidden_size, cudaStream_t stream,
    const __half *token_emb, const __half *pos_emb, const int *token_id,
    __half *output, int *real_seq_len, int padding_id);

template <typename T>
__global__ void ker_bias_gelu(T* input, const T* bias,
                   int feature_dim) {
  int offset = blockIdx.x * feature_dim;
  for(int idx = threadIdx.x; idx < feature_dim; idx += blockDim.x) {
      int cur_offset = offset + idx;
      input[cur_offset] = gelu<float>(
        input[cur_offset] + __ldg(&bias[idx]));
  }
}

template <>
__global__ void ker_bias_gelu<__half>(__half* input, const __half* bias,
                   int feature_dim) {  
  int offset = blockIdx.x * feature_dim;
  half2* pinput = (half2*) input;
  const half2* pbias = (const half2*) bias;
  for(int idx = threadIdx.x; idx < feature_dim; idx += blockDim.x) {
      int cur_offset = offset + idx;
      pinput[cur_offset] = gelu<half2>(
        __hadd2(pinput[cur_offset], __ldg(&pbias[idx])));
  }
}

template <typename T>
void ker_bias_gelu_launcher(
    int batch_token_num, int block_dim, cudaStream_t stream,
    T* input, const T* bias, int feature_dim) {
  ker_bias_gelu<T><<<batch_token_num,
      block_dim, 0, stream>>>(input, bias, feature_dim);
}

template <>
void ker_bias_gelu_launcher<__half>(
    int batch_token_num, int block_dim, cudaStream_t stream,
    __half* input, const __half* bias, int feature_dim) {
  ker_bias_gelu<__half><<<batch_token_num,
      block_dim, 0, stream>>>(input, bias, feature_dim / 2);
}

template void ker_bias_gelu_launcher<float>(
    int batch_token_num, int block_dim, cudaStream_t stream,
    float* input, const float* bias, int feature_dim);

template void ker_bias_gelu_launcher<__half>(
    int batch_token_num, int block_dim, cudaStream_t stream,
    __half* input, const __half* bias, int feature_dim);

template <typename T>
__global__ void ker_correlation_softmax_gpt(T* correlation,
                                            const int* real_seq_len) {
  /**
  @brief
  query-key correlation softmax for encoder self attention

  @thread
  gridDim.x = batch_size
  gridDim.y = head_num * batch_seq_len
  blockDim.x = batch_seq_len

  @param
  correlation: [batch_size, head_num, batch_seq_len, batch_seq_len]
  real_seq_len: [batch_size]
  */
  int query_token_pos = blockIdx.y % blockDim.x;
  if (query_token_pos >= real_seq_len[blockIdx.x]) {
    return;
  }
  int mask = 0;
  if (threadIdx.x > query_token_pos) {
    mask = 1; // Can only see the token on the left side of it
  }

  int idx = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
  float val = (float)correlation[idx];
  float max_val = blockReduceMax<float>(mask ? CUDA_FLOAT_INF_NEG : val);
  __shared__ float smax;
  if (threadIdx.x == 0) smax = max_val;
  __syncthreads();

  val = mask ? 0.f : expf(fmaxf(logit_thresh_min, val - smax));
  float rsum = blockReduceSum<float>(val);
  __shared__ float ssum;
  if (threadIdx.x == 0) ssum = rsum;
  __syncthreads();

  correlation[idx] = (T)(val / (ssum + epsilon));
}

template <typename T>
void ker_correlation_softmax_gpt_launcher(
    int batch_size, int batch_seq_len, int head_num, cudaStream_t stream,
    T* correlation, const int* real_seq_len) {
  ker_correlation_softmax_gpt<T><<<dim3(batch_size, head_num * batch_seq_len),
      batch_seq_len, 0, stream>>>(correlation, real_seq_len);
}

template void ker_correlation_softmax_gpt_launcher<float>(
    int batch_size, int batch_seq_len, int head_num, cudaStream_t stream,
    float* correlation, const int* real_seq_len);

template void ker_correlation_softmax_gpt_launcher<__half>(
    int batch_size, int batch_seq_len, int head_num, cudaStream_t stream,
    __half* correlation, const int* real_seq_len);

template <typename T>
__global__ void ker_ppl(const T* logits,
  const int* input_ids, const int* real_seq_len,
  float* ppl,
  int vocab_size) {
  /**
  @brief
  compute ppl from logit
  ppl = - (1 / n) * sum(log(i|i-1...))
  one thread block compute log probability for the given token

  @thread
  gridDim.x = batch_size
  gridDim.y = batch_seq_len
  blockDim.x = max_thread_per_block

  @param
  logits: [batch_size, batch_seq_len, vocab_size]
  input_ids: [batch_size, batch_seq_len]
  real_seq_len: [batch_size]
  ppl: [batch_size]
  */  
  int seq_len = real_seq_len[blockIdx.x] - 1; // remove "eos"
  if (blockIdx.y >= seq_len) {
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
  for(int idx = left_logit_idx; idx < right_logit_idx; idx += blockDim.x) {
    max_logit = fmaxf(max_logit, (float)logits[idx]);
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
  for(int idx = left_logit_idx; idx < right_logit_idx; idx += blockDim.x) {
    float lgt = fmaxf((float)logits[idx] - s_max_logit, logit_thresh_min);
    sum_exp_logit += expf(lgt);
  }
  sum_exp_logit = blockReduceSum(sum_exp_logit);

  if (threadIdx.x == 0) {
    int token_id = input_ids[token_idx_in_batch + 1];
    float log_prob =
      ((float)logits[token_idx_in_batch * vocab_size + token_id] -
      s_max_logit - logf(sum_exp_logit)) / (float)seq_len;
    atomicAdd(ppl + blockIdx.x, -log_prob);
  }
}

template <typename T>
void ker_ppl_launcher(
    int batch_size, int batch_seq_len, int max_thread_per_block, cudaStream_t stream,
    const T* logits, const int* input_ids, const int* real_seq_len, float* ppl,
    int vocab_size) {
  ker_ppl<T><<<dim3(batch_size, batch_seq_len),
      max_thread_per_block, 0, stream>>>(logits,
      input_ids, real_seq_len, ppl, vocab_size);
}

template void ker_ppl_launcher<float>(
    int batch_size, int batch_seq_len, int max_thread_per_block, cudaStream_t stream,
    const float* logits, const int* input_ids, const int* real_seq_len, float* ppl,
    int vocab_size);

template void ker_ppl_launcher<__half>(
    int batch_size, int batch_seq_len, int max_thread_per_block, cudaStream_t stream,
    const __half* logits, const int* input_ids, const int* real_seq_len, float* ppl,
    int vocab_size);

}  // namespace nmt
}  // namespace lab
