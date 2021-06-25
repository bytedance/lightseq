#include "common.h"
#include "transformerKernels.h"

/**
@file
Implemented the cuda kernel function and its launcher
that required by transformer model.
Currently, fp16 and fp32 versions are provided
*/
namespace lightseq {
namespace cuda {

/**
@brief: select_beam_rough_topk
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
__global__ void select_beam_rough_topk(
    const T* logits, const T* logit_bias, const float* seq_probs,
    const float* seq_score, const int* alive_seq, int* can_idx,
    float* can_score, int* num_beam_can, int vocab_size, int max_step,
    float length_norm, int cur_step, float diverse_lambda, int end_id) {
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
    float lgt = (float)logits[i] + (float)__ldg(&logit_bias[i - block_start]);
    rough_top_kth_logit = fmaxf(rough_top_kth_logit, lgt);
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
    if (vocab_id < vocab_size) {
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
void select_beam_rough_topk_launcher(
    const T* logits, const T* logit_bias, const float* seq_probs,
    const float* seq_score, const int* alive_seq, int* can_idx,
    float* can_score, int* num_beam_can, int vocab_size, int max_step,
    float length_norm, int cur_step, int step_token_num,
    int max_thread_per_block, cudaStream_t stream, int beam_size,
    float diverse_lambda, int end_id) {
  if (beam_size == 1)
    select_beam_rough_topk<T, 1>
        <<<step_token_num, max_thread_per_block, 0, stream>>>(
            logits, logit_bias, seq_probs, seq_score, alive_seq, can_idx,
            can_score, num_beam_can, vocab_size, max_step, length_norm,
            cur_step, diverse_lambda, end_id);
  if (beam_size == 2)
    select_beam_rough_topk<T, 2>
        <<<step_token_num, max_thread_per_block, 0, stream>>>(
            logits, logit_bias, seq_probs, seq_score, alive_seq, can_idx,
            can_score, num_beam_can, vocab_size, max_step, length_norm,
            cur_step, diverse_lambda, end_id);
  if (beam_size == 4)
    select_beam_rough_topk<T, 4>
        <<<step_token_num, max_thread_per_block, 0, stream>>>(
            logits, logit_bias, seq_probs, seq_score, alive_seq, can_idx,
            can_score, num_beam_can, vocab_size, max_step, length_norm,
            cur_step, diverse_lambda, end_id);
  if (beam_size == 8)
    select_beam_rough_topk<T, 8>
        <<<step_token_num, max_thread_per_block, 0, stream>>>(
            logits, logit_bias, seq_probs, seq_score, alive_seq, can_idx,
            can_score, num_beam_can, vocab_size, max_step, length_norm,
            cur_step, diverse_lambda, end_id);
  if (beam_size == 16)
    select_beam_rough_topk<T, 16>
        <<<step_token_num, max_thread_per_block, 0, stream>>>(
            logits, logit_bias, seq_probs, seq_score, alive_seq, can_idx,
            can_score, num_beam_can, vocab_size, max_step, length_norm,
            cur_step, diverse_lambda, end_id);
  if (beam_size == 32)
    select_beam_rough_topk<T, 32>
        <<<step_token_num, max_thread_per_block, 0, stream>>>(
            logits, logit_bias, seq_probs, seq_score, alive_seq, can_idx,
            can_score, num_beam_can, vocab_size, max_step, length_norm,
            cur_step, diverse_lambda, end_id);
}

template void select_beam_rough_topk_launcher<float>(
    const float* logits, const float* logit_bias, const float* seq_probs,
    const float* seq_score, const int* alive_seq, int* can_idx,
    float* can_score, int* num_beam_can, int vocab_size, int max_step,
    float length_norm, int cur_step, int step_token_num,
    int max_thread_per_block, cudaStream_t stream, int beam_size,
    float diverse_lambda, int end_id);

template void select_beam_rough_topk_launcher<__half>(
    const __half* logits, const __half* logit_bias, const float* seq_probs,
    const float* seq_score, const int* alive_seq, int* can_idx,
    float* can_score, int* num_beam_can, int vocab_size, int max_step,
    float length_norm, int cur_step, int step_token_num,
    int max_thread_per_block, cudaStream_t stream, int beam_size,
    float diverse_lambda, int end_id);

/**
@brief: ker_diverse_beam_search
Add different diverse score to can_score in each beam

@thread
gridDim.x = batch_size * beam_size
blockDim.x = max_thread_per_block

@param
can_score: [batch_size * beam_size * candidates_size] candidates_size is
dynamic
can_ids: [batch_size * beam_size * candidates_size]
num_beam_can: [1 + batch_size * beam_size]
*/
__global__ void ker_diverse_beam_search(float* can_score, int* can_ids,
                                        int* num_beam_can, int beam_size,
                                        float diverse_lambda, int vocab_size) {
  int total_candidates = num_beam_can[0];
  num_beam_can += 1;
  int can_pos = num_beam_can[blockIdx.x];
  int batch_id = blockIdx.x / beam_size;
  int beam_score_left_idx = can_pos + threadIdx.x;
  int beam_score_right_idx = blockIdx.x == (gridDim.x - 1)
                                 ? total_candidates
                                 : num_beam_can[blockIdx.x + 1];
  for (int idx = beam_score_left_idx; idx < beam_score_right_idx;
       idx += blockDim.x) {
    atomicAdd(can_score + idx, batch_id * min_log_probability -
                                   min_log_probability * blockIdx.x -
                                   diverse_lambda * (idx - can_pos + 1));
    int ori_can_idx = can_ids[idx];  // can_beam_id * vocab_size + vocab_id
    int can_beam_id = ori_can_idx / vocab_size;
    int can_vocab_id = ori_can_idx % vocab_size;
    can_ids[idx] =
        (can_beam_id + (idx - can_pos) * beam_size) * vocab_size + can_vocab_id;
  }
}

void ker_diverse_beam_search_launcher(float* can_score, int* can_ids,
                                      int* num_beam_can, int step_token_num,
                                      int max_thread_per_block,
                                      cudaStream_t stream, int beam_size,
                                      float diverse_lambda, int vocab_size) {
  ker_diverse_beam_search<<<step_token_num, max_thread_per_block, 0, stream>>>(
      can_score, can_ids, num_beam_can, beam_size, diverse_lambda, vocab_size);
}

/**
@brief: ker_bias_relu
add bias, activated by relu

@thread
gridDim.x = batch_size * batch_seq_len
blockDim.x = max_thread_per_block

@param
input: [batch_size * batch_seq_len, feature_dim]
bias: [feature_dim]
feature_dim: the dim of input feature
*/
template <typename T>
__global__ void ker_bias_relu(T* input, const T* bias, int feature_dim) {
  int offset = blockIdx.x * feature_dim;
  for (int idx = threadIdx.x; idx < feature_dim; idx += blockDim.x) {
    int cur_offset = offset + idx;
    input[cur_offset] = max(input[cur_offset] + __ldg(&bias[idx]), (T)0.f);
  }
}

template <>
__global__ void ker_bias_relu<__half>(__half* input, const __half* bias,
                                      int feature_dim) {
  int offset = blockIdx.x * feature_dim;
  half2* pinput = (half2*)input;
  const half2* pbias = (const half2*)bias;
  for (int idx = threadIdx.x; idx < feature_dim; idx += blockDim.x) {
    int cur_offset = offset + idx;
    float2 f2_inp = __half22float2(pinput[cur_offset]);
    float2 f2_bias = __half22float2(__ldg(&pbias[idx]));
    f2_inp.x = fmaxf(f2_inp.x + f2_bias.x, 0.f);
    f2_inp.y = fmaxf(f2_inp.y + f2_bias.y, 0.f);
    pinput[cur_offset] = __float22half2_rn(f2_inp);
  }
}

template <typename T>
void ker_bias_relu_launcher(int batch_token_num, int block_dim,
                            cudaStream_t stream, T* input, const T* bias,
                            int feature_dim) {
  ker_bias_relu<T>
      <<<batch_token_num, block_dim, 0, stream>>>(input, bias, feature_dim);
}

template <>
void ker_bias_relu_launcher<__half>(int batch_token_num, int block_dim,
                                    cudaStream_t stream, __half* input,
                                    const __half* bias, int feature_dim) {
  ker_bias_relu<__half>
      <<<batch_token_num, block_dim, 0, stream>>>(input, bias, feature_dim / 2);
}

template void ker_bias_relu_launcher<float>(int batch_token_num, int block_dim,
                                            cudaStream_t stream, float* input,
                                            const float* bias, int feature_dim);

template void ker_bias_relu_launcher<__half>(int batch_token_num, int block_dim,
                                             cudaStream_t stream, __half* input,
                                             const __half* bias,
                                             int feature_dim);
/**
@brief: ker_norm_layer
layer normalization

@thread
gridDim.x = batch_size * batch_seq_len
blockDim.x = max_thread_per_block

@param
matrix: [batch_size, batch_seq_len, hidden_size]
scale: [hidden_size]
bias: [hidden_size]
*/
template <typename T>
__global__ void ker_norm_layer(T* matrix, const T* scale, const T* bias,
                               int hidden_size) {
  uint block_start = blockIdx.x * hidden_size;
  uint start = block_start + threadIdx.x;
  uint end = block_start + hidden_size;
  float val = 0.0;
  for (uint i = start; i < end; i += blockDim.x) {
    val += matrix[i];
  }

  // step 0. compute mean
  __shared__ float s_mean;
  float reduce_res = blockReduceSum<float>(val);
  if (threadIdx.x == 0) s_mean = reduce_res / float(hidden_size);
  __syncthreads();

  // step 1. compute variance
  val = 0.0;
  for (uint i = start; i < end; i += blockDim.x) {
    float tmp = matrix[i] - s_mean;
    val += tmp * tmp;
  }
  __shared__ float s_var;
  reduce_res = blockReduceSum(val);
  if (threadIdx.x == 0)
    s_var = rsqrtf(reduce_res / float(hidden_size) + epsilon);
  __syncthreads();

  // step 2. layer norm
  for (uint i = start; i < end; i += blockDim.x) {
    val = matrix[i] - s_mean;
    matrix[i] = val * s_var * __ldg(&scale[i - block_start]) +
                __ldg(&bias[i - block_start]);
  }
}

template <>
__global__ void ker_norm_layer<__half>(__half* matrix, const __half* scale,
                                       const __half* bias,
                                       int half_hidden_size) {
  uint block_start = blockIdx.x * half_hidden_size;
  uint start = block_start + threadIdx.x;
  uint end = blockIdx.x * half_hidden_size + half_hidden_size;
  half2* pmatrix = (half2*)matrix;
  const half2* pscale = (const half2*)scale;
  const half2* pbias = (const half2*)bias;
  float mean_dim = float(half_hidden_size) * 2.f;

  float val = 0.0;
  // step 0. compute mean
  for (uint i = start; i < end; i += blockDim.x) {
    float2 local_f2 = safe_half2_to_float2(pmatrix[i]);
    val += local_f2.x + local_f2.y;
  }
  __shared__ float s_mean;
  float reduce_res = blockReduceSum<float>(val);
  if (threadIdx.x == 0) s_mean = reduce_res / mean_dim;
  __syncthreads();

  // step 1. compute variance
  val = 0.0;
  for (uint i = start; i < end; i += blockDim.x) {
    float2 local_f2 = safe_half2_to_float2(pmatrix[i]);
    float tmpx = local_f2.x - s_mean;
    float tmpy = local_f2.y - s_mean;
    val += tmpx * tmpx + tmpy * tmpy;
  }
  __shared__ float s_var;
  reduce_res = blockReduceSum(val);
  if (threadIdx.x == 0) s_var = rsqrtf(reduce_res / mean_dim + epsilon);
  __syncthreads();

  // step 2. layer norm
  for (uint i = start; i < end; i += blockDim.x) {
    float2 scale_val = __half22float2(__ldg(&pscale[i - block_start]));
    float2 bias_val = __half22float2(__ldg(&pbias[i - block_start]));
    float2 local_f2 = safe_half2_to_float2(pmatrix[i]);
    local_f2.x = (local_f2.x - s_mean) * s_var * scale_val.x + bias_val.x;
    local_f2.y = (local_f2.y - s_mean) * s_var * scale_val.y + bias_val.y;
    pmatrix[i] = __float22half2_rn(local_f2);
  }
}

template <typename T>
void ker_norm_layer_launcher(int token_num, int hidden_size,
                             cudaStream_t stream, T* matrix, const T* scale,
                             const T* bias, int max_thread_per_block) {
  ker_norm_layer<T><<<token_num, max_thread_per_block, 0, stream>>>(
      matrix, scale, bias, hidden_size);
}

template <>
void ker_norm_layer_launcher<__half>(int token_num, int hidden_size,
                                     cudaStream_t stream, __half* matrix,
                                     const __half* scale, const __half* bias,
                                     int max_thread_per_block) {
  ker_norm_layer<__half><<<token_num, max_thread_per_block, 0, stream>>>(
      matrix, scale, bias, hidden_size / 2);
}

template void ker_norm_layer_launcher<float>(int token_num, int hidden_size,
                                             cudaStream_t stream, float* matrix,
                                             const float* scale,
                                             const float* bias,
                                             int max_thread_per_block);

template void ker_norm_layer_launcher<__half>(
    int token_num, int hidden_size, cudaStream_t stream, __half* matrix,
    const __half* scale, const __half* bias, int max_thread_per_block);
/**
@brief: ker_norm_layer_resual
layer normalization, and add an residual_bias to input

@thread
gridDim.x = batch_size * batch_seq_len
blockDim.x = max_thread_per_block

@param
matrix: [batch_size, batch_seq_len, hidden_size]
scale: [hidden_size]
bias: [hidden_size]
residual_bias: [hidden_size]
*/
template <typename T>
__global__ void ker_norm_layer_resual(T* input, T* output, const T* scale,
                                      const T* bias, const T* residual_bias,
                                      const int hidden_size, bool is_post_ln) {
  uint block_start = blockIdx.x * hidden_size;
  uint start = block_start + threadIdx.x;
  uint end = block_start + hidden_size;
  float val = 0.0;
  for (uint i = start; i < end; i += blockDim.x) {
    val += input[i];
  }

  // step 0. compute mean
  __shared__ float s_mean;
  float reduce_res = blockReduceSum<float>(val);
  if (threadIdx.x == 0) s_mean = reduce_res / float(hidden_size);
  __syncthreads();

  // step 1. compute variance
  val = 0.0;
  for (uint i = start; i < end; i += blockDim.x) {
    float tmp = input[i] - s_mean;
    val += tmp * tmp;
  }
  __shared__ float s_var;
  reduce_res = blockReduceSum(val);
  if (threadIdx.x == 0)
    s_var = rsqrtf(reduce_res / float(hidden_size) + epsilon);
  __syncthreads();

  // step 2. layer norm
  for (uint i = start; i < end; i += blockDim.x) {
    val = input[i] - s_mean;
    output[i] = val * s_var * __ldg(&scale[i - block_start]) +
                __ldg(&bias[i - block_start]);
    if (is_post_ln) {
      input[i] = output[i] + __ldg(&residual_bias[i - block_start]);
    } else {
      input[i] += __ldg(&residual_bias[i - block_start]);
    }
  }
}

template <>
__global__ void ker_norm_layer_resual<__half>(
    __half* input, __half* output, const __half* scale, const __half* bias,
    const __half* residual_bias, const int half_hidden_size, bool is_post_ln) {
  uint block_start = blockIdx.x * half_hidden_size;
  uint start = block_start + threadIdx.x;
  uint end = blockIdx.x * half_hidden_size + half_hidden_size;
  half2* pinput = (half2*)input;
  half2* poutput = (half2*)output;
  const half2* pscale = (const half2*)scale;
  const half2* pbias = (const half2*)bias;
  const half2* presidual_bias = (const half2*)residual_bias;
  float mean_dim = float(half_hidden_size) * 2.f;

  float val = 0.0;
  // step 0. compute mean
  for (uint i = start; i < end; i += blockDim.x) {
    float2 local_f2 = safe_half2_to_float2(pinput[i]);
    val += local_f2.x + local_f2.y;
  }
  __shared__ float s_mean;
  float reduce_res = blockReduceSum<float>(val);
  if (threadIdx.x == 0) s_mean = reduce_res / mean_dim;
  __syncthreads();

  // step 1. compute variance
  val = 0.0;
  for (uint i = start; i < end; i += blockDim.x) {
    float2 local_f2 = safe_half2_to_float2(pinput[i]);
    float tmpx = local_f2.x - s_mean;
    float tmpy = local_f2.y - s_mean;
    val += tmpx * tmpx + tmpy * tmpy;
  }
  __shared__ float s_var;
  reduce_res = blockReduceSum(val);
  if (threadIdx.x == 0) s_var = rsqrtf(reduce_res / mean_dim + epsilon);
  __syncthreads();

  // step 2. layer norm
  for (uint i = start; i < end; i += blockDim.x) {
    float2 scale_val = __half22float2(__ldg(&pscale[i - block_start]));
    float2 bias_val = __half22float2(__ldg(&pbias[i - block_start]));
    float2 local_f2 = safe_half2_to_float2(pinput[i]);
    local_f2.x = (local_f2.x - s_mean) * s_var * scale_val.x + bias_val.x;
    local_f2.y = (local_f2.y - s_mean) * s_var * scale_val.y + bias_val.y;
    poutput[i] = __float22half2_rn(local_f2);
    if (!is_post_ln) {
      local_f2 = safe_half2_to_float2(pinput[i]);
    }
    float2 residual_bias_val =
        __half22float2(__ldg(&presidual_bias[i - block_start]));
    float2 new_input_f2;
    new_input_f2.x = local_f2.x + residual_bias_val.x;
    new_input_f2.y = local_f2.y + residual_bias_val.y;
    pinput[i] = __float22half2_rn(new_input_f2);
  }
}

template <typename T>
void ker_norm_layer_resual_launcher(int token_num, int hidden_size,
                                    cudaStream_t stream, T* input, T* output,
                                    const T* scale, const T* bias,
                                    const T* residual_bias,
                                    const int max_thread_per_block,
                                    bool is_post_ln) {
  ker_norm_layer_resual<T><<<token_num, max_thread_per_block, 0, stream>>>(
      input, output, scale, bias, residual_bias, hidden_size, is_post_ln);
}

template <>
void ker_norm_layer_resual_launcher<__half>(int token_num, int hidden_size,
                                            cudaStream_t stream, __half* input,
                                            __half* output, const __half* scale,
                                            const __half* bias,
                                            const __half* residual_bias,
                                            const int max_thread_per_block,
                                            bool is_post_ln) {
  ker_norm_layer_resual<__half><<<token_num, max_thread_per_block, 0, stream>>>(
      input, output, scale, bias, residual_bias, hidden_size / 2, is_post_ln);
}

template void ker_norm_layer_resual_launcher<float>(
    int token_num, int hidden_size, cudaStream_t stream, float* input,
    float* output, const float* scale, const float* bias,
    const float* residual_bias, const int max_thread_per_block,
    bool is_post_ln);

template void ker_norm_layer_resual_launcher<__half>(
    int token_num, int hidden_size, cudaStream_t stream, __half* input,
    __half* output, const __half* scale, const __half* bias,
    const __half* residual_bias, const int max_thread_per_block,
    bool is_post_ln);

/**
@brief: ker_enc_embedding
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
__global__ void ker_enc_embedding(const T* token_emb, const T* pos_emb,
                                  const int* token_id, T* output,
                                  int* padding_mask, int padding_id,
                                  const int hidden_size) {
  int target_pos = blockIdx.x * gridDim.y + blockIdx.y;
  int start = target_pos * hidden_size + threadIdx.x;
  int end = (target_pos + 1) * hidden_size;
  int tid = token_id[target_pos];
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
                pos_emb[blockIdx.y * hidden_size + offset];
  }
}

template <>
__global__ void ker_enc_embedding<__half>(const __half* token_emb,
                                          const __half* pos_emb,
                                          const int* token_id, __half* output,
                                          int* padding_mask, int padding_id,
                                          const int half_hidden_size) {
  int target_pos = blockIdx.x * gridDim.y + blockIdx.y;
  int start = target_pos * half_hidden_size + threadIdx.x;
  int end = (target_pos + 1) * half_hidden_size;
  int tid = token_id[target_pos];
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
    te.x += pe.x;
    te.y += pe.y;
    output_h[i] = __float22half2_rn(te);
  }
}

template <typename T>
void ker_enc_embedding_launcher(int batch_size, int batch_seq_len,
                                int hidden_size, cudaStream_t stream,
                                const T* token_emb, const T* pos_emb,
                                const int* token_id, T* output,
                                int* padding_mask, int padding_id,
                                int max_thread_per_block) {
  ker_enc_embedding<T>
      <<<dim3(batch_size, batch_seq_len), max_thread_per_block, 0, stream>>>(
          token_emb, pos_emb, token_id, output, padding_mask, padding_id,
          hidden_size);
}

template <>
void ker_enc_embedding_launcher<__half>(int batch_size, int batch_seq_len,
                                        int hidden_size, cudaStream_t stream,
                                        const __half* token_emb,
                                        const __half* pos_emb,
                                        const int* token_id, __half* output,
                                        int* padding_mask, int padding_id,
                                        int max_thread_per_block) {
  ker_enc_embedding<__half>
      <<<dim3(batch_size, batch_seq_len), max_thread_per_block, 0, stream>>>(
          token_emb, pos_emb, token_id, output, padding_mask, padding_id,
          hidden_size / 2);
}

template void ker_enc_embedding_launcher<float>(
    int batch_size, int batch_seq_len, int hidden_size, cudaStream_t stream,
    const float* token_emb, const float* pos_emb, const int* token_id,
    float* output, int* padding_mask, int padding_id, int max_thread_per_block);

template void ker_enc_embedding_launcher<__half>(
    int batch_size, int batch_seq_len, int hidden_size, cudaStream_t stream,
    const __half* token_emb, const __half* pos_emb, const int* token_id,
    __half* output, int* padding_mask, int padding_id,
    int max_thread_per_block);

/**
@brief: ker_dec_embedding
for decoder, look up token embedding, add position embedding

@thread
gridDim.x = batch_size * beam_size
blockDim.x = max_thread_per_block

@param
token_emb: [hidden_size, vocab_size], note, it is different with encoder
pos_emb: [max_step, hidden_size]
token_id: input token id, [batch_size, beam_size, max_step]
output: result, [batch_size, beam_size, hidden_size]
step: current step
max_step: max decoder steps
vocab_size: vocabulary size
*/
template <typename T>
__global__ void ker_dec_embedding(const T* token_emb, const T* pos_emb,
                                  const int* token_id, T* output, int step,
                                  int max_step, int vocab_size,
                                  int hidden_size) {
  for (uint offset = threadIdx.x; offset < hidden_size; offset += blockDim.x) {
    int token_idx = token_id[blockIdx.x * max_step + step];
    output[blockIdx.x * hidden_size + offset] =
        token_emb[offset * vocab_size + token_idx] +
        pos_emb[step * hidden_size + offset];
  }
}

template <typename T>
void ker_dec_embedding_launcher(int step_token_num, int hidden_size,
                                cudaStream_t stream, const T* token_emb,
                                const T* pos_emb, const int* token_id,
                                T* output, int step, int max_step,
                                int vocab_size, int max_thread_per_block) {
  ker_dec_embedding<T><<<step_token_num, max_thread_per_block, 0, stream>>>(
      token_emb, pos_emb, token_id, output, step, max_step, vocab_size,
      hidden_size);
}

template void ker_dec_embedding_launcher<float>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const float* token_emb, const float* pos_emb, const int* token_id,
    float* output, int step, int max_step, int vocab_size,
    int max_thread_per_block);

template void ker_dec_embedding_launcher<__half>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const __half* token_emb, const __half* pos_emb, const int* token_id,
    __half* output, int step, int max_step, int vocab_size,
    int max_thread_per_block);

/**
@brief: ker_arrange_encself_qkv
split and reshape ori_qkv matrix into new_q, new_k, new_v during encoder
self-attention
ori_qkv is the result of gemm

@thread
gridDim.x = batch_size * batch_seq_len
gridDim.y = 3
blockDim.x = max_thread_per_block

@param
ori_qkv: [batch_size, batch_seq_len, 3, hidden_size]
qkv_bias: [3, hidden_size]
new_qkv: [3, batch_size, head_num, batch_seq_len, dim_per_head]
max_batch_dim: max_batch_size * max_seq_len * hidden_size
batch_seq_len: the sequence length of the current batch
dim_per_head: dim of one head in multi-head attention
head_num: head number in multi-head attention
*/
template <typename T>
__global__ void ker_arrange_encself_qkv(const T* ori_qkv, const T* qkv_bias,
                                        T* new_qkv, int max_batch_dim,
                                        int batch_seq_len, int dim_per_head,
                                        int head_num) {
  int hidden_size = dim_per_head * head_num;
  int batch_id = blockIdx.x / batch_seq_len;
  int token_id = blockIdx.x % batch_seq_len;
  int qkv_offset = max_batch_dim * blockIdx.y;
  for (std::size_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    int target_id = targetid_4dim(batch_id, head_id, token_id, dim_id, head_num,
                                  batch_seq_len, dim_per_head);
    new_qkv[qkv_offset + target_id] =
        ori_qkv[(blockIdx.x * gridDim.y + blockIdx.y) * hidden_size + i] +
        __ldg(&qkv_bias[blockIdx.y * hidden_size + i]);
  }
}

template <>
__global__ void ker_arrange_encself_qkv<__half>(
    const __half* ori_qkv, const __half* qkv_bias, __half* new_qkv,
    int max_batch_dim, int batch_seq_len, int dim_per_head, int head_num) {
  int hidden_size = dim_per_head * head_num;
  int batch_id = blockIdx.x / batch_seq_len;
  int token_id = blockIdx.x % batch_seq_len;
  for (std::size_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    int qkv_offset = max_batch_dim * blockIdx.y;
    int target_id = targetid_4dim(batch_id, head_id, token_id, dim_id, head_num,
                                  batch_seq_len, dim_per_head);

    const half2* p_ori_qkv = (const half2*)ori_qkv;
    const half2* p_bias = (const half2*)qkv_bias;
    half2* p_new_qkv = (half2*)new_qkv;
    p_new_qkv[qkv_offset + target_id] = __hadd2(
        p_ori_qkv[(blockIdx.x * gridDim.y + blockIdx.y) * hidden_size + i],
        __ldg(&p_bias[blockIdx.y * hidden_size + i]));
  }
}

template <typename T>
void ker_arrange_encself_qkv_launcher(int batch_token_num, int hidden_size,
                                      cudaStream_t stream, const T* ori_qkv,
                                      const T* qkv_bias, T* new_qkv,
                                      int max_batch_dim, int batch_seq_len,
                                      int dim_per_head, int head_num,
                                      int max_thread_per_block) {
  ker_arrange_encself_qkv<T>
      <<<dim3(batch_token_num, 3), max_thread_per_block, 0, stream>>>(
          ori_qkv, qkv_bias, new_qkv, max_batch_dim, batch_seq_len,
          dim_per_head, head_num);
}

template <>
void ker_arrange_encself_qkv_launcher<__half>(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const __half* ori_qkv, const __half* qkv_bias, __half* new_qkv,
    int max_batch_dim, int batch_seq_len, int dim_per_head, int head_num,
    int max_thread_per_block) {
  ker_arrange_encself_qkv<__half>
      <<<dim3(batch_token_num, 3), max_thread_per_block, 0, stream>>>(
          ori_qkv, qkv_bias, new_qkv, max_batch_dim / 2, batch_seq_len,
          dim_per_head / 2, head_num);
}

template void ker_arrange_encself_qkv_launcher<float>(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const float* ori_qkv, const float* qkv_bias, float* new_qkv,
    int max_batch_dim, int batch_seq_len, int dim_per_head, int head_num,
    int max_thread_per_block);

template void ker_arrange_encself_qkv_launcher<__half>(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const __half* ori_qkv, const __half* qkv_bias, __half* new_qkv,
    int max_batch_dim, int batch_seq_len, int dim_per_head, int head_num,
    int max_thread_per_block);

/**
@brief: ker_arrange_decself_qkv
split and reshape ori_qkv matrix into new_q, new_k, new_v during decoder
self-attention
ori_qkv is the result of gemm

@thread
gridDim.x = batch_size * beam_size
gridDim.y = 3
blockDim.x = max_thread_per_block

@param
ori_qkv: [batch_size, beam_size, 3, hidden_size]
qkv_bias: [3, hidden_size]
new_q: new query. [batch_size, beam_size, hidden_size]
new_k: new key. [batch_size, beam_size, head_num, max_step, dim_per_head]
new_v: new value. [batch_size, beam_size, head_num, max_step, dim_per_head]
head_num: head number in multi-head attention
dim_per_head: dim of one head in multi-head attention
max_step: max decode step
step_id: current step id
*/
template <typename T>
__global__ void ker_arrange_decself_qkv(const T* ori_qkv, const T* qkv_bias,
                                        T* new_q, T* new_k, T* new_v,
                                        int head_num, int dim_per_head,
                                        int max_step, int step_id) {
  int hidden_size = dim_per_head * head_num;
  for (std::size_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    // blockdim is equal to hidden_size
    T val = ori_qkv[(blockIdx.x * gridDim.y + blockIdx.y) * hidden_size + i] +
            __ldg(&qkv_bias[blockIdx.y * hidden_size + i]);
    int seq_id =
        blockIdx.x;  // obvious， seq_id = batch_id * beam_size + beam_id
    if (blockIdx.y == 0) {
      // for query
      new_q[seq_id * hidden_size + i] = val;
      return;
    }
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    int target_id = targetid_4dim(seq_id, head_id, step_id, dim_id, head_num,
                                  max_step, dim_per_head);
    if (blockIdx.y == 1) {
      // for key
      new_k[target_id] = val;
    } else {
      // for value
      new_v[target_id] = val;
    }
  }
}

template <>
__global__ void ker_arrange_decself_qkv<__half>(
    const __half* ori_qkv, const __half* qkv_bias, __half* new_q, __half* new_k,
    __half* new_v, int head_num, int dim_per_head, int max_step, int step_id) {
  int half_hidden_size = dim_per_head * head_num;
  const half2* p_qkv = (const half2*)ori_qkv;
  const half2* p_bias = (const half2*)qkv_bias;
  for (std::size_t i = threadIdx.x; i < half_hidden_size; i += blockDim.x) {
    half2 val = __hadd2(
        p_qkv[(blockIdx.x * gridDim.y + blockIdx.y) * half_hidden_size + i],
        __ldg(&p_bias[blockIdx.y * half_hidden_size + i]));
    // obvious，seq_id = batch_id * beam_size + beam_id
    int seq_id = blockIdx.x;
    if (blockIdx.y == 0) {
      // for query
      ((half2*)new_q)[seq_id * half_hidden_size + i] = val;
      return;
    }
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    int target_id = targetid_4dim(seq_id, head_id, step_id, dim_id, head_num,
                                  max_step, dim_per_head);
    if (blockIdx.y == 1) {
      // for key
      ((half2*)new_k)[target_id] = val;
    } else {
      // for value
      ((half2*)new_v)[target_id] = val;
    }
  }
}

template <typename T>
void ker_arrange_decself_qkv_launcher(int step_token_num, int hidden_size,
                                      cudaStream_t stream, const T* ori_qkv,
                                      const T* qkv_bias, T* new_q, T* new_k,
                                      T* new_v, int head_num, int dim_per_head,
                                      int max_step, int step_id,
                                      int max_thread_per_block) {
  ker_arrange_decself_qkv<T>
      <<<dim3(step_token_num, 3), max_thread_per_block, 0, stream>>>(
          ori_qkv, qkv_bias, new_q, new_k, new_v, head_num, dim_per_head,
          max_step, step_id);
}

template <>
void ker_arrange_decself_qkv_launcher<__half>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const __half* ori_qkv, const __half* qkv_bias, __half* new_q, __half* new_k,
    __half* new_v, int head_num, int dim_per_head, int max_step, int step_id,
    int max_thread_per_block) {
  ker_arrange_decself_qkv<__half>
      <<<dim3(step_token_num, 3), max_thread_per_block, 0, stream>>>(
          ori_qkv, qkv_bias, new_q, new_k, new_v, head_num, dim_per_head / 2,
          max_step, step_id);
}

template void ker_arrange_decself_qkv_launcher<float>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const float* ori_qkv, const float* qkv_bias, float* new_q, float* new_k,
    float* new_v, int head_num, int dim_per_head, int max_step, int step_id,
    int max_thread_per_block);

template void ker_arrange_decself_qkv_launcher<__half>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const __half* ori_qkv, const __half* qkv_bias, __half* new_q, __half* new_k,
    __half* new_v, int head_num, int dim_per_head, int max_step, int step_id,
    int max_thread_per_block);

/**
@brief: ker_arrange_encdec_kv
split and reshape ori_kv matrix into new_k, new_v before enc-dec attention
it will be call once on encoder ouput

@thread
gridDim.x = batch_size * batch_seq_len
gridDim.y = dec_layer_num * 2
blockDim.x = max_thread_per_block

@param
ori_kv: [batch_size, batch_seq_len, dec_layer_num, 2, hidden_size]
kv_bias: [dec_layer_num, 2, hidden_size]
new_k: [batch_size, head_num, batch_seq_len, dim_per_head] per layer,
  with an offset in offset_per_layer between layers.
new_v: [batch_size, head_num, batch_seq_len, dim_per_head] per layer,
  with an offset in offset_per_layer between layers.
offset_per_layer: max_batch_size * max_step * hidden_size
batch_seq_len: sequence length of current batch
dim_per_head: dim of one head in multi-head attention
head_num: head number in multi-head attention
*/
template <typename T>
__global__ void ker_arrange_encdec_kv(const T* ori_kv, const T* kv_bias,
                                      T* new_k, T* new_v, int offset_per_layer,
                                      int batch_seq_len, int dim_per_head,
                                      int head_num) {
  int hidden_size = dim_per_head * head_num;
  for (std::size_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    T val = ori_kv[(blockIdx.x * gridDim.y + blockIdx.y) * hidden_size + i] +
            __ldg(&kv_bias[blockIdx.y * hidden_size + i]);
    int seq_id = blockIdx.x / batch_seq_len;
    int token_id = blockIdx.x % batch_seq_len;
    int layer_id = blockIdx.y >> 1;
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    int layer_offset = layer_id * offset_per_layer;
    int target_id = targetid_4dim(seq_id, head_id, token_id, dim_id, head_num,
                                  batch_seq_len, dim_per_head) +
                    layer_offset;

    if (blockIdx.y & 1) {
      // for value
      new_v[target_id] = val;
    } else {
      // for key
      new_k[target_id] = val;
    }
  }
}

template <>
__global__ void ker_arrange_encdec_kv<__half>(
    const __half* ori_kv, const __half* kv_bias, __half* new_k, __half* new_v,
    int offset_per_layer, int batch_seq_len, int dim_per_head, int head_num) {
  int half_hidden_size = dim_per_head * head_num;
  for (std::size_t i = threadIdx.x; i < half_hidden_size; i += blockDim.x) {
    const half2* p_ori_kv = (const half2*)ori_kv;
    const half2* p_kv_bias = (const half2*)kv_bias;
    half2 val = __hadd2(
        p_ori_kv[(blockIdx.x * gridDim.y + blockIdx.y) * half_hidden_size + i],
        __ldg(&p_kv_bias[blockIdx.y * half_hidden_size + i]));
    int seq_id = blockIdx.x / batch_seq_len;
    int token_id = blockIdx.x % batch_seq_len;
    int layer_id = blockIdx.y >> 1;
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    int layer_offset = layer_id * offset_per_layer;
    int target_id = targetid_4dim(seq_id, head_id, token_id, dim_id, head_num,
                                  batch_seq_len, dim_per_head) +
                    layer_offset;

    if (blockIdx.y & 1) {
      // for value
      ((half2*)new_v)[target_id] = val;
    } else {
      // for key
      ((half2*)new_k)[target_id] = val;
    }
  }
}

template <typename T>
void ker_arrange_encdec_kv_launcher(int batch_token_num, int dec_layer_num,
                                    int hidden_size, cudaStream_t stream,
                                    const T* ori_kv, const T* kv_bias, T* new_k,
                                    T* new_v, int offset_per_layer,
                                    int batch_seq_len, int dim_per_head,
                                    int head_num, int max_thread_per_block) {
  ker_arrange_encdec_kv<T>
      <<<dim3(batch_token_num, dec_layer_num * 2), max_thread_per_block, 0,
         stream>>>(ori_kv, kv_bias, new_k, new_v, offset_per_layer,
                   batch_seq_len, dim_per_head, head_num);
}

template <>
void ker_arrange_encdec_kv_launcher<__half>(
    int batch_token_num, int dec_layer_num, int hidden_size,
    cudaStream_t stream, const __half* ori_kv, const __half* kv_bias,
    __half* new_k, __half* new_v, int offset_per_layer, int batch_seq_len,
    int dim_per_head, int head_num, int max_thread_per_block) {
  ker_arrange_encdec_kv<__half>
      <<<dim3(batch_token_num, dec_layer_num * 2), max_thread_per_block, 0,
         stream>>>(ori_kv, kv_bias, new_k, new_v, offset_per_layer / 2,
                   batch_seq_len, dim_per_head / 2, head_num);
}

template void ker_arrange_encdec_kv_launcher<float>(
    int batch_token_num, int dec_layer_num, int hidden_size,
    cudaStream_t stream, const float* ori_kv, const float* kv_bias,
    float* new_k, float* new_v, int offset_per_layer, int batch_seq_len,
    int dim_per_head, int head_num, int max_thread_per_block);

template void ker_arrange_encdec_kv_launcher<__half>(
    int batch_token_num, int dec_layer_num, int hidden_size,
    cudaStream_t stream, const __half* ori_kv, const __half* kv_bias,
    __half* new_k, __half* new_v, int offset_per_layer, int batch_seq_len,
    int dim_per_head, int head_num, int max_thread_per_block);

/**
@brief: ker_arrange_encdec_q
reshape ori_q into new_q and add bias
during enc-dec attention
ori_q is the result of gemm

@thread
gridDim.x = batch_size * beam_size
blockDim.x = max_thread_per_block

@param
ori_q: [batch_size, beam_size, hidden_size]
q_bias: [hidden_size]
new_q: [batch_size, head_num, beam_size, dim_per_head]
beam_size: beam size of beam search
dim_per_head: dim of one head in multi-head attention
head_num: head number in multi-head attention
*/
template <typename T>
__global__ void ker_arrange_encdec_q(const T* ori_q, const T* q_bias, T* new_q,
                                     int beam_size, int dim_per_head,
                                     int head_num) {
  int hidden_size = dim_per_head * head_num;
  for (std::size_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    T val = ori_q[blockIdx.x * hidden_size + i] + __ldg(&q_bias[i]);
    int batch_id = blockIdx.x / beam_size;
    int beam_id = blockIdx.x % beam_size;
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    new_q[targetid_4dim(batch_id, head_id, beam_id, dim_id, head_num, beam_size,
                        dim_per_head)] = val;
  }
}

template <>
__global__ void ker_arrange_encdec_q<__half>(const __half* ori_q,
                                             const __half* q_bias,
                                             __half* new_q, int beam_size,
                                             int dim_per_head, int head_num) {
  int half_hidden_size = dim_per_head * head_num;
  for (std::size_t i = threadIdx.x; i < half_hidden_size; i += blockDim.x) {
    const half2* p_q = (const half2*)ori_q;
    const half2* p_bias = (const half2*)q_bias;
    half2 val =
        __hadd2(p_q[blockIdx.x * half_hidden_size + i], __ldg(&p_bias[i]));
    int batch_id = blockIdx.x / beam_size;
    int beam_id = blockIdx.x % beam_size;
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    ((half2*)new_q)[targetid_4dim(batch_id, head_id, beam_id, dim_id, head_num,
                                  beam_size, dim_per_head)] = val;
  }
}

template <typename T>
void ker_arrange_encdec_q_launcher(int step_token_num, int hidden_size,
                                   cudaStream_t stream, const T* ori_q,
                                   const T* q_bias, T* new_q, int beam_size,
                                   int dim_per_head, int head_num,
                                   int max_thread_per_block) {
  ker_arrange_encdec_q<T><<<step_token_num, max_thread_per_block, 0, stream>>>(
      ori_q, q_bias, new_q, beam_size, dim_per_head, head_num);
}

template <>
void ker_arrange_encdec_q_launcher<__half>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const __half* ori_q, const __half* q_bias, __half* new_q, int beam_size,
    int dim_per_head, int head_num, int max_thread_per_block) {
  ker_arrange_encdec_q<__half>
      <<<step_token_num, max_thread_per_block, 0, stream>>>(
          ori_q, q_bias, new_q, beam_size, dim_per_head / 2, head_num);
}

template void ker_arrange_encdec_q_launcher<float>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const float* ori_q, const float* q_bias, float* new_q, int beam_size,
    int dim_per_head, int head_num, int max_thread_per_block);

template void ker_arrange_encdec_q_launcher<__half>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const __half* ori_q, const __half* q_bias, __half* new_q, int beam_size,
    int dim_per_head, int head_num, int max_thread_per_block);

/**
@brief: ker_correlation_softmax_encself
query-key correlation softmax for encoder self attention

@thread
gridDim.x = batch_size
gridDim.y = head_num * batch_seq_len
blockDim.x = first multiple of WARP_SIZE greater than batch_seq_len

@param
correlation: [batch_size, head_num, batch_seq_len, batch_seq_len]
src_padding_mask: [batch_size, batch_seq_len],
  indicating which token is a padding token.
*/
template <typename T>
__global__ void ker_correlation_softmax_encself(T* correlation,
                                                const int* src_padding_mask,
                                                int batch_seq_len) {
  int idx = (blockIdx.x * gridDim.y + blockIdx.y) * batch_seq_len + threadIdx.x;
  if (threadIdx.x < batch_seq_len &&
      src_padding_mask[blockIdx.x * batch_seq_len +
                       blockIdx.y % batch_seq_len]) {
    correlation[idx] = (T)0.f;
    return;
  }
  int mask = threadIdx.x < batch_seq_len
                 ? src_padding_mask[blockIdx.x * batch_seq_len + threadIdx.x]
                 : 1;
  float val = threadIdx.x < batch_seq_len ? (float)correlation[idx]
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

  if (threadIdx.x < batch_seq_len) correlation[idx] = (T)(val / ssum);
}

template <typename T>
void ker_correlation_softmax_encself_launcher(int batch_size, int batch_seq_len,
                                              int head_num, cudaStream_t stream,
                                              T* correlation,
                                              const int* src_padding_mask) {
  int block_dim = batch_seq_len;
  if (batch_seq_len < 1024) {
    block_dim = (batch_seq_len + 31) >> 5;
    block_dim *= 32;
  }

  ker_correlation_softmax_encself<T>
      <<<dim3(batch_size, head_num * batch_seq_len), block_dim, 0, stream>>>(
          correlation, src_padding_mask, batch_seq_len);
}

template void ker_correlation_softmax_encself_launcher<float>(
    int batch_size, int batch_seq_len, int head_num, cudaStream_t stream,
    float* correlation, const int* src_padding_mask);

template void ker_correlation_softmax_encself_launcher<__half>(
    int batch_size, int batch_seq_len, int head_num, cudaStream_t stream,
    __half* correlation, const int* src_padding_mask);

/**
@brief: ker_correlation_softmax_decself
query-key correlation softmax for decoder self attention

@thread
gridDim.x = batch_size * beam_size * head_num
blockDim.x = first multiple of WARP_SIZE greater than cur_step + 1

@param
correlation: [batch_size, beam_size, head_num, cur_step + 1]
*/
template <typename T>
__global__ void ker_correlation_softmax_decself(T* correlation, int step_num) {
  int idx = blockIdx.x * step_num + threadIdx.x;
  float val =
      threadIdx.x < step_num ? (float)correlation[idx] : CUDA_FLOAT_INF_NEG;

  float max_val = blockReduceMax(val);
  __shared__ float smax;
  if (threadIdx.x == 0) smax = max_val;
  __syncthreads();

  val = threadIdx.x < step_num ? expf(val - smax) : 0;

  float rsum = blockReduceSum(val);
  __shared__ float ssum;
  if (threadIdx.x == 0) ssum = rsum;
  __syncthreads();

  if (threadIdx.x < step_num) correlation[idx] = (T)(val / ssum);
}

template <typename T>
void ker_correlation_softmax_decself_launcher(int batch_head_num, int step_num,
                                              cudaStream_t stream,
                                              T* correlation) {
  int block_dim = step_num;
  if (step_num < 1024) {
    block_dim = (step_num + 31) >> 5;
    block_dim *= 32;
  }
  ker_correlation_softmax_decself<<<batch_head_num, block_dim, 0, stream>>>(
      correlation, step_num);
}

template void ker_correlation_softmax_decself_launcher<float>(
    int batch_head_num, int step_num, cudaStream_t stream, float* correlation);

template void ker_correlation_softmax_decself_launcher<__half>(
    int batch_head_num, int step_num, cudaStream_t stream, __half* correlation);

/**
@brief: ker_correlation_softmax_encdec
query-key correlation softmax for encoder-decoder attention

@thread
gridDim.x = batch_size
gridDim.y = head_num * beam_size
blockDim.x = first multiple of WARP_SIZE greater than batch_seq_len

@param
correlation: [batch_size, head_num, beam_size, batch_seq_len]
src_padding_mask: [batch_size, batch_seq_len]
  indicating which token is a padding token.
*/
template <typename T>
__global__ void ker_correlation_softmax_encdec(T* correlation,
                                               const int* src_padding_mask,
                                               int batch_seq_len) {
  int idx = (blockIdx.x * gridDim.y + blockIdx.y) * batch_seq_len + threadIdx.x;
  int mask = threadIdx.x < batch_seq_len
                 ? src_padding_mask[blockIdx.x * batch_seq_len + threadIdx.x]
                 : 1;
  float val = threadIdx.x < batch_seq_len ? (float)correlation[idx]
                                          : CUDA_FLOAT_INF_NEG;

  float max_val = blockReduceMax(mask ? CUDA_FLOAT_INF_NEG : val);
  __shared__ float smax;
  if (threadIdx.x == 0) smax = max_val;
  __syncthreads();

  val = mask ? 0.f : expf(val - smax);
  float rsum = blockReduceSum(val);
  __shared__ float ssum;
  if (threadIdx.x == 0) ssum = rsum;
  __syncthreads();

  if (threadIdx.x < batch_seq_len) correlation[idx] = (T)(val / ssum);
}

template <typename T>
void ker_correlation_softmax_encdec_launcher(
    int batch_size, int head_num_per_seq, int batch_seq_len,
    cudaStream_t stream, T* correlation, const int* src_padding_mask) {
  int block_dim = batch_seq_len;
  if (batch_seq_len < 1024) {
    block_dim = (batch_seq_len + 31) >> 5;
    block_dim *= 32;
  }
  ker_correlation_softmax_encdec<T>
      <<<dim3(batch_size, head_num_per_seq), block_dim, 0, stream>>>(
          correlation, src_padding_mask, batch_seq_len);
}

template void ker_correlation_softmax_encdec_launcher<float>(
    int batch_size, int head_num_per_seq, int batch_seq_len,
    cudaStream_t stream, float* correlation, const int* src_padding_mask);

template void ker_correlation_softmax_encdec_launcher<__half>(
    int batch_size, int head_num_per_seq, int batch_seq_len,
    cudaStream_t stream, __half* correlation, const int* src_padding_mask);

/**
@brief: ker_arrange_atten_output
reshape Scaled Dot-Product Attention output.
It will be used by both encoder and decoder
token_num = batch_seq_len, for encoder
          = beam_size, for decoder

@thread
gridDim.x = batch_size * ${token_num}
blockDim.x = max_thread_per_block

@param
ori_q: [batch_size, head_num, ${token_num}, dim_per_head]
new_q: [batch_size, ${token_num}, hidden_size]
beam_size : for decoder, beam_size is beam_size; for encoder, beam_size is
    batch_seq_len
dim_per_head: dim of one head in multi-head attention
head_num: head number in multi-head attention
*/
template <typename T>
__global__ void ker_arrange_atten_output(const T* ori_q, T* new_q,
                                         int beam_size, int dim_per_head,
                                         int head_num) {
  int hidden_size = dim_per_head * head_num;
  int batch_id = blockIdx.x / beam_size;
  // note, for encoder, beam_id is token_id; for decoder, beam_id is beam_id
  int beam_id = blockIdx.x % beam_size;
  for (std::size_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    new_q[blockIdx.x * hidden_size + i] = ori_q[targetid_4dim(
        batch_id, head_id, beam_id, dim_id, head_num, beam_size, dim_per_head)];
  }
}

template <>
__global__ void ker_arrange_atten_output<__half>(const __half* ori_q,
                                                 __half* new_q, int beam_size,
                                                 int dim_per_head,
                                                 int head_num) {
  int batch_id = blockIdx.x / beam_size;
  // note, for encoder, beam_id is token_id; for decoder, beam_id is beam_id
  int beam_id = blockIdx.x % beam_size;
  int half_hidden_size = dim_per_head * head_num;
  for (std::size_t i = threadIdx.x; i < half_hidden_size; i += blockDim.x) {
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;
    const half2* p_ori_q = (const half2*)ori_q;
    half2* p_new_q = (half2*)new_q;
    p_new_q[blockIdx.x * half_hidden_size + i] = p_ori_q[targetid_4dim(
        batch_id, head_id, beam_id, dim_id, head_num, beam_size, dim_per_head)];
  }
}

template <typename T>
void ker_arrange_atten_output_launcher(int batch_token_num, int hidden_size,
                                       cudaStream_t stream, const T* ori_q,
                                       T* new_q, int beam_size,
                                       int dim_per_head, int head_num,
                                       int max_thread_per_block) {
  ker_arrange_atten_output<T>
      <<<batch_token_num, max_thread_per_block, 0, stream>>>(
          ori_q, new_q, beam_size, dim_per_head, head_num);
}

template <>
void ker_arrange_atten_output_launcher<__half>(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const __half* ori_q, __half* new_q, int beam_size, int dim_per_head,
    int head_num, int max_thread_per_block) {
  ker_arrange_atten_output<__half>
      <<<batch_token_num, max_thread_per_block, 0, stream>>>(
          ori_q, new_q, beam_size, dim_per_head / 2, head_num);
}

template void ker_arrange_atten_output_launcher<float>(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const float* ori_q, float* new_q, int beam_size, int dim_per_head,
    int head_num, int max_thread_per_block);

template void ker_arrange_atten_output_launcher<__half>(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const __half* ori_q, __half* new_q, int beam_size, int dim_per_head,
    int head_num, int max_thread_per_block);

/**
@brief: ker_refresh_result
refresh alive_seq, seq_probs, seq_score, num_finish_beam based on
sorted candidate

@thread
gridDim.x = batch_size
gridDim.y = beam_size
blockDim.x = max_step

@param
can_idx: [none], no certain length, determined by rough candidate number
can_score: [none], no certain length, determined by rough candidate number
num_can_per_beam: [batch_size * beam_size]
    save exclusive_scan_sum of the beam candidate number array
    e.g. [0,2,5,1] -> [0, 0, 2, 7]
old_alive_seq: [batch_size, beam_size, max_step]
new_alive_seq: [batch_size, beam_size, max_step]
seq_probs: [batch_size, beam_size]
seq_score: [batch_size, beam_size]
    performing length penlty on seq_probs will get seq_probs
num_finish_beam: record current finished beam.
    it will be use to decide whether early stop during beam_search
vocab_size: target vocabulary size
cur_step: current step
length_norm: length penlty norm value
*/
__global__ void ker_refresh_result(const int* can_idx, const float* can_score,
                                   const int* num_can_per_beam,
                                   const int* old_alive_seq, int* new_alive_seq,
                                   float* seq_probs, float* seq_score,
                                   int* num_finish_beam, int vocab_size,
                                   int cur_step, float length_norm,
                                   float diverse_lambda, int end_id) {
  // step1 update alive_seq
  int can_pos = num_can_per_beam[blockIdx.x * gridDim.y] + blockIdx.y;
  int ori_can_idx = can_idx[can_pos];  // can_beam_id * vocab_size + vocab_id
  int can_beam_id = ori_can_idx / vocab_size;
  int can_vocab_id = ori_can_idx % vocab_size;
  int rank_id;
  if (diverse_lambda != 0) {
    rank_id = can_beam_id / gridDim.y;  // rank in each beam
    can_beam_id %= gridDim.y;
  }
  int thread_vocab_id;
  if (threadIdx.x > cur_step + 1) {
    thread_vocab_id = end_id;
  } else if (threadIdx.x == cur_step + 1) {
    // add current step generate vocabulary id
    thread_vocab_id = can_vocab_id;
  } else {
    // threadIdx.x <= cur_step
    thread_vocab_id = old_alive_seq[targetid_3dim(
        blockIdx.x, can_beam_id, threadIdx.x, gridDim.y, blockDim.x)];
  }
  new_alive_seq[targetid_3dim(blockIdx.x, blockIdx.y, threadIdx.x, gridDim.y,
                              blockDim.x)] = thread_vocab_id;

  // step2 update seq_probs if alive seq when not eos
  if (cur_step == 0 || can_vocab_id != end_id) {
    // alive seq
    if (threadIdx.x == 0) {
      if (diverse_lambda == 0) {
        seq_probs[blockIdx.x * gridDim.y + blockIdx.y] =
            (can_score[can_pos] - blockIdx.x * min_log_probability) /
            length_norm;  // recover it
      } else {
        seq_probs[blockIdx.x * gridDim.y + blockIdx.y] =
            (can_score[can_pos] - blockIdx.x * min_log_probability +
             diverse_lambda * (rank_id + 1)) /
            length_norm;
      }
    }
    return;
  }

  // step3 update seq_score, num_finish_beam if finish seq
  if (threadIdx.x == 0) {
    atomicAdd(num_finish_beam, 1);
  }
  int seq_last_id = old_alive_seq[targetid_3dim(
      blockIdx.x, can_beam_id, cur_step, gridDim.y, blockDim.x)];
  // update finished seq score
  if (threadIdx.x == 0) {
    // note, with batch offset value, to sort between batch element
    if (diverse_lambda == 0) {
      seq_score[blockIdx.x * gridDim.y + blockIdx.y] = can_score[can_pos];
    } else {
      seq_score[blockIdx.x * gridDim.y + blockIdx.y] =
          can_score[can_pos] + diverse_lambda * (rank_id + 1);
    }
  }
}  // namespace cuda

/**
@brief: ker_refresh_cache
supply current step's projected k,v to K, V cache

@thread
gridDim.x = decoder_layer_num * (step_id + 1)
gridDim.y = batch_size * beam_size * 2
blockDim.x = max_thread_per_block

@param
num_can_per_beam: [batch_size, beam_size]
can_idx: [none], no certain length, determined by rough candidate number
self_k_bgeem: [batch_size, beam_size, head_num, max_step, dim_per_head] *
    decoder_layer_num
self_v_bgeem: [batch_size, beam_size, head_num, max_step, dim_per_head] *
    decoder_layer_num
new_self_k_bgeem: [batch_size, beam_size, head_num, max_step, dim_per_head] *
    decoder_layer_num
new_self_v_bgeem: [batch_size, beam_size, head_num, max_step, dim_per_head] *
    decoder_layer_num
self_k_bgeem_offset = max_batch_size * max_step * hidden_size * beam_size
beam_size : beam size for beam_search
dim_per_head: dim of one head in multi-head attention
head_num: head number in multi-head attention
vocab_size: the vocab size of decoder
cur_step: current step
max_step: max decode step
*/
template <typename T>
__global__ void ker_refresh_cache(const int* num_can_per_beam,
                                  const int* can_idx, const T* self_k_bgeem,
                                  const T* self_v_bgeem, T* new_self_k_bgeem,
                                  T* new_self_v_bgeem, int self_k_bgeem_offset,
                                  int beam_size, int dim_per_head, int head_num,
                                  int vocab_size, int cur_step, int max_step,
                                  bool diverse, int end_id) {
  int layer_id = blockIdx.x / (cur_step + 1);
  int step_id = blockIdx.x % (cur_step + 1);
  int kv_id = blockIdx.y & 1;
  int beam_id_global = blockIdx.y >> 1;
  int batch_id = beam_id_global / beam_size;
  int beam_id = beam_id_global % beam_size;
  int hidden_size = dim_per_head * head_num;
  for (std::size_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;

    int can_pos = num_can_per_beam[batch_id * beam_size] + beam_id;
    int can_beam_id =
        can_idx[can_pos] / vocab_size;  // can_beam_id * vocab_size + vocab_id
    if (diverse) can_beam_id %= beam_size;
    if (can_idx[can_pos] % vocab_size == end_id) {
      return;
    }

    int base_pos = targetid_5dim(batch_id, 0, head_id, step_id, dim_id,
                                 beam_size, head_num, max_step, dim_per_head) +
                   layer_id * self_k_bgeem_offset;
    int beam_offset = hidden_size * max_step;
    int ori_id = base_pos + beam_offset * can_beam_id;
    int new_id = base_pos + beam_offset * beam_id;
    if (kv_id == 0) {
      // for key
      new_self_k_bgeem[new_id] = self_k_bgeem[ori_id];
    } else {
      // for value
      new_self_v_bgeem[new_id] = self_v_bgeem[ori_id];
    }
  }
}

template <>
__global__ void ker_refresh_cache<__half>(
    const int* num_can_per_beam, const int* can_idx, const __half* self_k_bgeem,
    const __half* self_v_bgeem, __half* new_self_k_bgeem,
    __half* new_self_v_bgeem, int self_k_bgeem_offset, int beam_size,
    int dim_per_head, int head_num, int vocab_size, int cur_step, int max_step,
    bool diverse, int end_id) {
  int layer_id = blockIdx.x / (cur_step + 1);
  int step_id = blockIdx.x % (cur_step + 1);
  int kv_id = blockIdx.y & 1;
  int beam_id_global = blockIdx.y >> 1;
  int batch_id = beam_id_global / beam_size;
  int beam_id = beam_id_global % beam_size;
  int half_hidden_size = dim_per_head * head_num;
  for (std::size_t i = threadIdx.x; i < half_hidden_size; i += blockDim.x) {
    int head_id = i / dim_per_head;
    int dim_id = i % dim_per_head;

    int can_pos = num_can_per_beam[batch_id * beam_size] + beam_id;
    int can_beam_id =
        can_idx[can_pos] / vocab_size;  // can_beam_id * vocab_size + vocab_id
    if (diverse) can_beam_id %= beam_size;
    if (cur_step != 0 && can_idx[can_pos] % vocab_size == end_id) {
      return;
    }

    int base_pos = targetid_5dim(batch_id, 0, head_id, step_id, dim_id,
                                 beam_size, head_num, max_step, dim_per_head) +
                   layer_id * self_k_bgeem_offset;
    int beam_offset = half_hidden_size * max_step;
    int ori_id = base_pos + beam_offset * can_beam_id;
    int new_id = base_pos + beam_offset * beam_id;
    if (kv_id == 0) {
      // for key
      ((half2*)new_self_k_bgeem)[new_id] = ((half2*)self_k_bgeem)[ori_id];
    } else {
      // for value
      ((half2*)new_self_v_bgeem)[new_id] = ((half2*)self_v_bgeem)[ori_id];
    }
  }
}

template <typename T>
void ker_refresh_cache_launcher(
    int grid_dim_x, int grid_dim_y, int block_dim, cudaStream_t stream,
    const int* num_can_per_beam, const int* can_idx, const T* self_k_bgeem,
    const T* self_v_bgeem, T* new_self_k_bgeem, T* new_self_v_bgeem,
    int self_k_bgeem_offset, int beam_size, int dim_per_head, int head_num,
    int vocab_size, int cur_step, int max_step, bool diverse, int end_id) {
  ker_refresh_cache<T><<<dim3(grid_dim_x, grid_dim_y), block_dim, 0, stream>>>(
      num_can_per_beam, can_idx, self_k_bgeem, self_v_bgeem, new_self_k_bgeem,
      new_self_v_bgeem, self_k_bgeem_offset, beam_size, dim_per_head, head_num,
      vocab_size, cur_step, max_step, diverse, end_id);
}

template <>
void ker_refresh_cache_launcher<__half>(
    int grid_dim_x, int grid_dim_y, int block_dim, cudaStream_t stream,
    const int* num_can_per_beam, const int* can_idx, const __half* self_k_bgeem,
    const __half* self_v_bgeem, __half* new_self_k_bgeem,
    __half* new_self_v_bgeem, int self_k_bgeem_offset, int beam_size,
    int dim_per_head, int head_num, int vocab_size, int cur_step, int max_step,
    bool diverse, int end_id) {
  ker_refresh_cache<__half>
      <<<dim3(grid_dim_x, grid_dim_y), block_dim / 2, 0, stream>>>(
          num_can_per_beam, can_idx, self_k_bgeem, self_v_bgeem,
          new_self_k_bgeem, new_self_v_bgeem, self_k_bgeem_offset / 2,
          beam_size, dim_per_head / 2, head_num, vocab_size, cur_step, max_step,
          diverse, end_id);
}

template void ker_refresh_cache_launcher<float>(
    int grid_dim_x, int grid_dim_y, int block_dim, cudaStream_t stream,
    const int* num_can_per_beam, const int* can_idx, const float* self_k_bgeem,
    const float* self_v_bgeem, float* new_self_k_bgeem, float* new_self_v_bgeem,
    int self_k_bgeem_offset, int beam_size, int dim_per_head, int head_num,
    int vocab_size, int cur_step, int max_step, bool diverse, int end_id);

template void ker_refresh_cache_launcher<__half>(
    int grid_dim_x, int grid_dim_y, int block_dim, cudaStream_t stream,
    const int* num_can_per_beam, const int* can_idx, const __half* self_k_bgeem,
    const __half* self_v_bgeem, __half* new_self_k_bgeem,
    __half* new_self_v_bgeem, int self_k_bgeem_offset, int beam_size,
    int dim_per_head, int head_num, int vocab_size, int cur_step, int max_step,
    bool diverse, int end_id);

/**
@brief: ker_write_trg_tokenid_pos_penalty
write result from alive seq to output, for length_penlty >= 0
or length_penlty < 0 and decode to max_decode_step
simply output the beam0 as final result

@thread
gridDim.x = batch_size
blockDim.x = cur_step + 1

@param
alive_seq: [batch_size, beam_size, max_step], <start> is the first token in
each beam
output: [batch_size, cur_step + 1], no <start> and at least one <eos> in the
last of seq
*/
__global__ void ker_write_trg_tokenid_pos_penalty(const int* alive_seq,
                                                  float* seq_score, int* output,
                                                  int max_step, int beam_size) {
  int target_id =
      targetid_3dim(blockIdx.x, 0, threadIdx.x + 1, beam_size, max_step);
  output[blockIdx.x * blockDim.x + threadIdx.x] = alive_seq[target_id];
  if (threadIdx.x == 0) {
    seq_score[blockIdx.x] =
        seq_score[blockIdx.x * beam_size] - blockIdx.x * min_log_probability;
  }
}

/**
@brief: ker_write_trg_tokenid_neg_penalty
write result from alive seq to output,
for length_penlty < 0 and all beam has reach it's eos
compute each beam's score and select the top beam

@thread
gridDim.x = batch_size
blockDim.x = cur_step + 1

@param
alive_seq: [batch_size, beam_size, max_step], <start> is the first token in
each beam
seq_score: [batch_size, beam_size], the length_penlty < 0, seq_score is also
the sum_log_probs
output: [batch_size, cur_step + 1], no <start> and at least one <eos> in the
last of seq
*/
__global__ void ker_write_trg_tokenid_neg_penalty(const int* alive_seq,
                                                  const float* seq_score,
                                                  int* output, int max_step,
                                                  int beam_size, int vocab_size,
                                                  int end_id) {
  __shared__ float seq_final_score;
  __shared__ int res_beam_id;
  if (threadIdx.x == 0) {
    seq_final_score = CUDA_FLOAT_INF_NEG;
    res_beam_id = 0;
  }
  for (int beam_id = 0; beam_id < beam_size; beam_id++) {
    int target_id = targetid_3dim(blockIdx.x, beam_id, threadIdx.x + 1,
                                  beam_size, max_step);
    int seq_len =
        blockReduceSum(int(alive_seq[target_id] != end_id));  // compute seq len
    if (threadIdx.x == 0) {
      float cur_beam_score = seq_score[blockIdx.x * beam_size + beam_id] -
                             blockIdx.x * min_log_probability;  // recover prob
      cur_beam_score /= (float(seq_len) + epsilon);
      if (cur_beam_score > seq_final_score) {
        seq_final_score = cur_beam_score;
        res_beam_id = beam_id;
      }
    }
    __syncthreads();
  }
  int target_id = targetid_3dim(blockIdx.x, res_beam_id, threadIdx.x + 1,
                                beam_size, max_step);
  output[blockIdx.x * blockDim.x + threadIdx.x] = alive_seq[target_id];
  // output[blockIdx.x * blockDim.x + threadIdx.x] =
  // int(seq_final_score[threadIdx.x]);
}

/**
@brief: ker_write_topk_result
write result from alive seq to output, recover seq_score
for length_penlty > 0

@thread
gridDim.x = batch_size * beam_size
blockDim.x = cur_step + 1

@param
alive_seq: [batch_size, beam_size, max_step], <start> is the first token in
each beam
seq_score: [batch_size, beam_size]
seq_probs: [batch_size, beam_size]
output: [batch_size, cur_step + 1], no <start> and at least one <eos> in the
last of seq
*/
__global__ void ker_write_topk_result(const int* alive_seq, float* seq_score,
                                      int* res_seq, int vocab_size,
                                      int max_step, int beam_size, int end_id) {
  res_seq[blockIdx.x * blockDim.x + threadIdx.x] =
      alive_seq[blockIdx.x * max_step + threadIdx.x + 1];
  if (threadIdx.x == 0) {
    seq_score[blockIdx.x] -= (blockIdx.x / beam_size) * min_log_probability;
    res_seq[blockIdx.x * blockDim.x + blockDim.x - 1] = end_id;
  }
}

/**
@brief: ker_topk_sample
quick rough topk sampling from logits

@thread
gridDim.x = batch_size
blockDim.x = max_thread_per_block

@param
logits: [batch_size, logits_seq_len, vocab_size]
old_input_ids: [batch_size, batch_seq_len]
new_input_ids: [batch_size, batch_seq_len+1]
unfinished: [1]
curandstate: [batch_size]
*/
template <typename T, int k>
__global__ void ker_topk_sample(const T* logits, const T* logit_bias,
                                int* old_input_ids, int* new_input_ids,
                                const int vocab_size, const int max_step,
                                const int batch_seq_len, int logits_seq_len,
                                int* unfinished, curandState* curandstate,
                                int eos_id) {
  int last_token_idx_in_batch = blockIdx.x * max_step + batch_seq_len - 1;

  /* add EOS to end if last token is EOS */
  if (batch_seq_len > 1 && old_input_ids[last_token_idx_in_batch] == eos_id) {
    if (threadIdx.x == 0) {
      old_input_ids[last_token_idx_in_batch + 1] = eos_id;
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
    rough_top_kth_logit = fmaxf(
        rough_top_kth_logit,
        (float)(logits[idx]) +
            (float)__ldg(&logit_bias[idx - left_logit_idx + threadIdx.x]));
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
    // int test_num = 0;
    __shared__ float s_topk_exp_sum;
    for (int idx = left_logit_idx; idx < right_logit_idx; idx += blockDim.x) {
      float logit =
          (float)logits[idx] +
          (float)__ldg(&logit_bias[idx - left_logit_idx + threadIdx.x]);
      float logit_exp = expf(fmaxf(logit - s_max_logit, logit_thresh_min));
      // if (logit >= s_topk_logit) test_num++;
      if (logit >= s_topk_logit && logit_exp > topk_exp) {
        topk_exp = logit_exp;
        topk_tid = idx - left_logit_idx + threadIdx.x;
      }
    }

    // test_num = blockReduceSum(test_num);
    // __shared__ int s_test_num;
    // if (threadIdx.x == 0) {
    //   s_test_num = test_num;
    //   if (s_test_num != 1) printf("sample from top %d\n", s_test_num);
    //   // printf("sample from top %s", test_num);
    // }
    // __syncthreads();

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
      float logit =
          (float)logits[idx] +
          (float)__ldg(&logit_bias[idx - left_logit_idx + threadIdx.x]);
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

  /* step3 write back new sampled ids */
  if (threadIdx.x == 0) {
    old_input_ids[last_token_idx_in_batch + 1] = s_tid;
  }
}

template <typename T>
void ker_topk_sample_launcher(int batch_size, int batch_seq_len,
                              const int max_step, int logits_seq_len,
                              int max_thread_per_block, cudaStream_t stream,
                              const T* logits, const T* logit_bias,
                              int* old_input_ids, int* new_input_ids,
                              const int vocab_size, const int k,
                              int* unfinished, curandState* curandstate,
                              int eos_id) {
  if (k == 1)
    ker_topk_sample<T, 1><<<batch_size, max_thread_per_block, 0, stream>>>(
        logits, logit_bias, old_input_ids, new_input_ids, vocab_size, max_step,
        batch_seq_len, logits_seq_len, unfinished, curandstate, eos_id);
  else if (k == 2)
    ker_topk_sample<T, 2><<<batch_size, max_thread_per_block, 0, stream>>>(
        logits, logit_bias, old_input_ids, new_input_ids, vocab_size, max_step,
        batch_seq_len, logits_seq_len, unfinished, curandstate, eos_id);
  else if (k == 4)
    ker_topk_sample<T, 4><<<batch_size, max_thread_per_block, 0, stream>>>(
        logits, logit_bias, old_input_ids, new_input_ids, vocab_size, max_step,
        batch_seq_len, logits_seq_len, unfinished, curandstate, eos_id);
  else if (k == 8)
    ker_topk_sample<T, 8><<<batch_size, max_thread_per_block, 0, stream>>>(
        logits, logit_bias, old_input_ids, new_input_ids, vocab_size, max_step,
        batch_seq_len, logits_seq_len, unfinished, curandstate, eos_id);
  else if (k == 16)
    ker_topk_sample<T, 16><<<batch_size, max_thread_per_block, 0, stream>>>(
        logits, logit_bias, old_input_ids, new_input_ids, vocab_size, max_step,
        batch_seq_len, logits_seq_len, unfinished, curandstate, eos_id);
  else if (k == 32)
    ker_topk_sample<T, 32><<<batch_size, max_thread_per_block, 0, stream>>>(
        logits, logit_bias, old_input_ids, new_input_ids, vocab_size, max_step,
        batch_seq_len, logits_seq_len, unfinished, curandstate, eos_id);
  else {
    throw std::invalid_argument("topk argument should be in [1,2,4,8,16,32]");
  }
}

template void ker_topk_sample_launcher<float>(
    int batch_size, int batch_seq_len, const int max_step, int logits_seq_len,
    int max_thread_per_block, cudaStream_t stream, const float* logits,
    const float* logit_bias, int* old_input_ids, int* new_input_idx,
    const int vocab_size, const int k, int* unfinished,
    curandState* curandstate, int eos_id);

template void ker_topk_sample_launcher<__half>(
    int batch_size, int batch_seq_len, const int max_step, int logits_seq_len,
    int max_thread_per_block, cudaStream_t stream, const __half* logits,
    const __half* logit_bias, int* old_input_ids, int* new_input_idx,
    const int vocab_size, const int k, int* unfinished,
    curandState* curandstate, int eos_id);

/**
@brief: ker_topp_sample
quick rough topp sampling from logits

@thread
gridDim.x = batch_size
blockDim.x = max_thread_per_block

@param
logits: [batch_size, logits_seq_len, vocab_size]
old_input_ids: [batch_size, batch_seq_len]
new_input_ids: [batch_size, batch_seq_len+1]
unfinished: [1]
curandstate: [batch_size]
*/
template <typename T>
__global__ void ker_topp_sample(const T* logits, const T* logit_bias,
                                int* old_input_ids, int* new_input_ids,
                                const int vocab_size, const int max_step,
                                const int batch_seq_len, int logits_seq_len,
                                int* unfinished, float p,
                                curandState* curandstate, int eos_id) {
  int token_idx_in_batch = blockIdx.x * max_step + batch_seq_len - 1;

  /* add EOS to end if last token is EOS */
  if (batch_seq_len > 1 && old_input_ids[token_idx_in_batch] == eos_id) {
    if (threadIdx.x == 0) {
      old_input_ids[token_idx_in_batch + 1] = eos_id;
    }
    return;
  }
  int logits_token_idx_in_batch =
      blockIdx.x * logits_seq_len + logits_seq_len - 1;
  int left_logit_idx = logits_token_idx_in_batch * vocab_size + threadIdx.x;
  int right_logit_idx = (logits_token_idx_in_batch + 1) * vocab_size;

  /* step1. find max logit in each thread and sample from these probs with
   * nucleus sampling */
  __shared__ float s_max_logit;
  float max_logit = CUDA_FLOAT_INF_NEG;
  for (int idx = left_logit_idx; idx < right_logit_idx; idx += blockDim.x) {
    max_logit = fmaxf(max_logit, (float)logits[idx]) +
                (float)__ldg(&logit_bias[idx - left_logit_idx + threadIdx.x]);
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

  /* step2 hold one logit per thread which larger than Kth logit and sample
   * from them */
  float topk_exp_sum, topk_exp = CUDA_FLOAT_INF_NEG;
  int topk_tid = vocab_size;
  int test_num = 0;
  __shared__ float s_topk_exp_sum;
  for (int idx = left_logit_idx; idx < right_logit_idx; idx += blockDim.x) {
    float logit = (float)logits[idx] +
                  (float)__ldg(&logit_bias[idx - left_logit_idx + threadIdx.x]);
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

  /* step3 write back new sampled ids */
  if (threadIdx.x == 0) {
    old_input_ids[token_idx_in_batch + 1] = s_tid;
  }
}

template <typename T>
void ker_topp_sample_launcher(int batch_size, int batch_seq_len,
                              const int max_step, int logits_seq_len,
                              int max_thread_per_block, cudaStream_t stream,
                              const T* logits, const T* logit_bias,
                              int* old_input_ids, int* new_input_ids,
                              const int vocab_size, const float p,
                              int* unfinished, curandState* curandstate,
                              int eos_id) {
  ker_topp_sample<T><<<batch_size, max_thread_per_block, 0, stream>>>(
      logits, logit_bias, old_input_ids, new_input_ids, vocab_size, max_step,
      batch_seq_len, logits_seq_len, unfinished, p, curandstate, eos_id);
}

template void ker_topp_sample_launcher<float>(
    int batch_size, int batch_seq_len, const int max_step, int logits_seq_len,
    int max_thread_per_block, cudaStream_t stream, const float* logits,
    const float* logit_bias, int* old_input_ids, int* new_input_idx,
    const int vocab_size, const float p, int* unfinished,
    curandState* curandstate, int eos_id);

template void ker_topp_sample_launcher<__half>(
    int batch_size, int batch_seq_len, const int max_step, int logits_seq_len,
    int max_thread_per_block, cudaStream_t stream, const __half* logits,
    const __half* logit_bias, int* old_input_ids, int* new_input_idx,
    const int vocab_size, const float p, int* unfinished,
    curandState* curandstate, int eos_id);

/**
@brief: ker_bias_gelu
add bias, activated by gelu

@thread
gridDim.x = batch_size * batch_seq_len
blockDim.x = max_thread_per_block

@param
input: [batch_size * batch_seq_len, feature_dim]
bias: [feature_dim]
feature_dim: the dim of input feature
*/
template <typename T>
__global__ void ker_bias_gelu(T* input, const T* bias, int feature_dim) {
  int offset = blockIdx.x * feature_dim;
  for (int idx = threadIdx.x; idx < feature_dim; idx += blockDim.x) {
    int cur_offset = offset + idx;
    input[cur_offset] = gelu<float>(input[cur_offset] + __ldg(&bias[idx]));
  }
}

/* fp16 version */
template <>
__global__ void ker_bias_gelu<__half>(__half* input, const __half* bias,
                                      int feature_dim) {
  int offset = blockIdx.x * feature_dim;
  half2* pinput = (half2*)input;
  const half2* pbias = (const half2*)bias;
  for (int idx = threadIdx.x; idx < feature_dim; idx += blockDim.x) {
    int cur_offset = offset + idx;
    pinput[cur_offset] =
        gelu<half2>(__hadd2(pinput[cur_offset], __ldg(&pbias[idx])));
  }
}

template <typename T>
void ker_bias_gelu_launcher(int batch_token_num, int block_dim,
                            cudaStream_t stream, T* input, const T* bias,
                            int feature_dim) {
  ker_bias_gelu<T>
      <<<batch_token_num, block_dim, 0, stream>>>(input, bias, feature_dim);
}

template <>
void ker_bias_gelu_launcher<__half>(int batch_token_num, int block_dim,
                                    cudaStream_t stream, __half* input,
                                    const __half* bias, int feature_dim) {
  ker_bias_gelu<__half>
      <<<batch_token_num, block_dim, 0, stream>>>(input, bias, feature_dim / 2);
}

template void ker_bias_gelu_launcher<float>(int batch_token_num, int block_dim,
                                            cudaStream_t stream, float* input,
                                            const float* bias, int feature_dim);

template void ker_bias_gelu_launcher<__half>(int batch_token_num, int block_dim,
                                             cudaStream_t stream, __half* input,
                                             const __half* bias,
                                             int feature_dim);

__global__ void ker_curand_setup(curandState* state) {
  /* Each thread gets same seed, a different sequence
     number, no offset */
  curand_init(clock(), blockIdx.x, 0, &state[blockIdx.x]);
}

}  // namespace cuda
}  // namespace lightseq
