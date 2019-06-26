#include "src/custom/transformer/kernels/nmtKernels.h"

namespace lab {
namespace nmt {

__global__ void kerBiasRelu(float* first_output, const float* first_bias,
                            unsigned int inner_size) {
  unsigned int col_offset = blockIdx.y * blockDim.x + threadIdx.x;
  unsigned int offset = blockIdx.x * inner_size + col_offset;
  first_output[offset] =
      max(first_output[offset] + first_bias[col_offset], 0.f);
}

__global__ void kerBiasAdd(float* ffn_input, const float* second_out,
                           const float* second_bias) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  ffn_input[i] += second_out[i] + second_bias[threadIdx.x];  // bias and add
}

__global__ void ker_norm_layer(float* matrix, const float* scale,
                               const float* bias) {
  __shared__ float s_mean;
  __shared__ float s_var;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  float val = matrix[i];
  float reduce_res = blockReduceSum(val);
  if (threadIdx.x == 0) s_mean = reduce_res / float(blockDim.x);
  __syncthreads();
  val -= s_mean;
  reduce_res = blockReduceSum(val * val);
  if (threadIdx.x == 0) s_var = reduce_res / float(blockDim.x);
  __syncthreads();
  matrix[i] =
      (val * rsqrtf(s_var + epsilon)) * scale[threadIdx.x] + bias[threadIdx.x];
}

__global__ void ker_norm_layer2(const float* input, float* output,
                                const float* scale, const float* bias) {
  __shared__ float s_mean;
  __shared__ float s_var;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  float val = input[i];
  float reduce_res = blockReduceSum(val);
  if (threadIdx.x == 0) s_mean = reduce_res / float(blockDim.x);
  __syncthreads();
  val -= s_mean;
  reduce_res = blockReduceSum(val * val);
  if (threadIdx.x == 0) s_var = reduce_res / float(blockDim.x);
  __syncthreads();
  output[i] =
      (val * rsqrtf(s_var + epsilon)) * scale[threadIdx.x] + bias[threadIdx.x];
}

__global__ void ker_norm_layer3(float* input, float* output, const float* scale,
                                const float* bias, const float* residual_bias) {
  __shared__ float s_mean;
  __shared__ float s_var;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  float val = input[i];
  input[i] = val + residual_bias[threadIdx.x];
  float reduce_res = blockReduceSum(val);
  if (threadIdx.x == 0) s_mean = reduce_res / float(blockDim.x);
  __syncthreads();
  val -= s_mean;
  reduce_res = blockReduceSum(val * val);
  if (threadIdx.x == 0) s_var = reduce_res / float(blockDim.x);
  __syncthreads();
  output[i] =
      (val * rsqrtf(s_var + epsilon)) * scale[threadIdx.x] + bias[threadIdx.x];
}

__global__ void ker_enc_embedding(const float* token_emb, const float* pos_emb,
                                  const int* token_id, float* output,
                                  int* padding_mask, int padding_id) {
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
  padding_mask: record padding token, [batch_size, batch_seq_len]
  padding_id, the padding_id, default 0
  */

  int target_pos = blockIdx.x * gridDim.y + blockIdx.y;
  int tid = token_id[target_pos];
  if (tid == padding_id) {
    // for padding id
    if (threadIdx.x == 0) padding_mask[target_pos] = 1;
    output[target_pos * blockDim.x + threadIdx.x] = 0.f;
    return;
  }
  if (threadIdx.x == 0) {
    padding_mask[target_pos] = 0;
  }
  output[target_pos * blockDim.x + threadIdx.x] =
      token_emb[tid * blockDim.x + threadIdx.x] +
      pos_emb[blockIdx.y * blockDim.x + threadIdx.x];
}

__global__ void ker_dec_embedding(const float* token_emb, const float* pos_emb,
                                  const int* token_id, float* output, int step,
                                  int max_step, int vocab_size) {
  /**
  @brief
  for decoder, look up token embedding, add position embedding

  @thread
  gridDim.x = batch_size * beam_size
  blockDim.x = hidden_size

  @param
  token_emb: [hidden_size, vocab_size], note, it is different with encoder
  pos_emb: [max_step, hidden_size]
  token_id: input token id, [batch_size, beam_size, max_step]
  output: result, [batch_size, beam_size, hidden_size]
  step: current step
  max_step: max decoder steps
  vocab_size: vocabulary size
  */

  int token_idx = token_id[blockIdx.x * max_step + step];
  output[blockIdx.x * blockDim.x + threadIdx.x] =
      token_emb[threadIdx.x * vocab_size + token_idx] +
      pos_emb[step * blockDim.x + threadIdx.x];
}

__global__ void ker_arrange_encself_qkv(const float* ori_qkv,
                                        const float* qkv_bias, float* new_qkv,
                                        int max_batch_dim, int batch_seq_len,
                                        int dim_per_head, int head_num) {
  // ori_qkv: [batch_size, batch_seq_len, 3, hidden_size]
  // new_q: [batch_size, head_num, batch_seq_len, dim_per_head]
  // new_k: [batch_size, head_num, dim_per_head, batch_seq_len]
  // new_v: [batch_size, head_num, batch_seq_len, dim_per_head]
  float val = ori_qkv[(blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x +
                      threadIdx.x] +
              qkv_bias[blockIdx.y * blockDim.x + threadIdx.x];
  int batch_id = blockIdx.x / batch_seq_len;
  int token_id = blockIdx.x % batch_seq_len;
  int head_id = threadIdx.x / dim_per_head;
  int dim_id = threadIdx.x % dim_per_head;
  int qkv_offset = max_batch_dim * blockIdx.y;
  int target_id = (blockIdx.y == 1)
                      ? targetid_4dim(batch_id, head_id, dim_id, token_id,
                                      head_num, dim_per_head, batch_seq_len)
                      : targetid_4dim(batch_id, head_id, token_id, dim_id,
                                      head_num, batch_seq_len, dim_per_head);
  new_qkv[qkv_offset + target_id] = val;
}

__global__ void ker_arrange_decself_qkv(const float* ori_qkv,
                                        const float* qkv_bias, float* new_q,
                                        float* new_k, float* new_v,
                                        int head_num, int dim_per_head,
                                        int max_step, int step_id) {
  // ori_qkv: [batch_size, beam_size, 3, hidden_size]
  // new_q: [batch_size, beam_size, hidden_size]
  // new_k: [batch_size, beam_size, head_num, dim_per_head, max_step]
  // new_v: [batch_size, beam_size, head_num, max_step, dim_per_head]
  // blockdim is equal to hidden_size
  float val = ori_qkv[(blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x +
                      threadIdx.x] +
              qkv_bias[blockIdx.y * blockDim.x + threadIdx.x];
  int seq_id = blockIdx.x;  // obvious， seq_id = batch_id * beam_size + beam_id
  if (blockIdx.y == 0) {
    // for query
    new_q[seq_id * blockDim.x + threadIdx.x] = val;
    return;
  }
  int head_id = threadIdx.x / dim_per_head;
  int dim_id = threadIdx.x % dim_per_head;
  if (blockIdx.y == 1) {
    // for key
    new_k[targetid_4dim(seq_id, head_id, dim_id, step_id, head_num,
                        dim_per_head, max_step)] = val;
  } else {
    // for value
    new_v[targetid_4dim(seq_id, head_id, step_id, dim_id, head_num, max_step,
                        dim_per_head)] = val;
  }
}

__global__ void ker_arrange_encdec_kv(const float* ori_kv, const float* kv_bias,
                                      float* new_k, float* new_v,
                                      int offset_per_layer, int batch_seq_len,
                                      int dim_per_head, int head_num) {
  /**
  @brief
  reshape encoder output into k, v for encdec attention

  @thread
  gridDim.x = batch_token_num
  gridDim.y = decoder_layer_num * 2
  blockDim.x = hidden_size

  @param
  ori_kv: [batch_size, batch_seq_len, layer_num, 2, hidden_size]
  new_k: layer_num * [batch_size, head_num, dim_per_head, batch_seq_len]
  new_v: layer_num * [batch_size, head_num, batch_seq_len, dim_per_head]
  blockdim is equal to hidden_size
  one layer offset: max_batch_size * max_step * hidden_size
  */
  float val =
      ori_kv[(blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x] +
      kv_bias[blockIdx.y * blockDim.x + threadIdx.x];
  int seq_id = blockIdx.x / batch_seq_len;
  int token_id = blockIdx.x % batch_seq_len;
  int layer_id = blockIdx.y >> 1;
  int head_id = threadIdx.x / dim_per_head;
  int dim_id = threadIdx.x % dim_per_head;
  int layer_offset = layer_id * offset_per_layer;

  if (blockIdx.y & 1) {
    // for value
    new_v[targetid_4dim(seq_id, head_id, token_id, dim_id, head_num,
                        batch_seq_len, dim_per_head) +
          layer_offset] = val;
  } else {
    // for key
    new_k[targetid_4dim(seq_id, head_id, dim_id, token_id, head_num,
                        dim_per_head, batch_seq_len) +
          layer_offset] = val;
  }
}

__global__ void ker_arrange_encdec_q(const float* ori_q, const float* q_bias,
                                     float* new_q, int beam_size,
                                     int dim_per_head, int head_num) {
  // ori_q: [batch_size, beam_size, hidden_size]
  // new_q: [batch_size, head_num, beam_size, dim_per_head]
  // blockdim is equal to hidden_size
  float val =
      ori_q[blockIdx.x * blockDim.x + threadIdx.x] + q_bias[threadIdx.x];
  int batch_id = blockIdx.x / beam_size;
  int beam_id = blockIdx.x % beam_size;
  int head_id = threadIdx.x / dim_per_head;
  int dim_id = threadIdx.x % dim_per_head;
  new_q[targetid_4dim(batch_id, head_id, beam_id, dim_id, head_num, beam_size,
                      dim_per_head)] = val;
}

__global__ void ker_correlation_softmax_encself(float* correlation,
                                                const int* src_padding_mask) {  
  /**
  @brief
  query-key correlation softmax for encoder self attention

  @thread
  gridDim.x = batch_size
  gridDim.y = head_num * batch_seq_len
  blockDim.x = batch_seq_len

  @param
  correlation: [batch_size, head_num, batch_seq_len, batch_seq_len]
  src_padding_mask: [batch_size, batch_seq_len]
  */
  if (src_padding_mask[blockIdx.x * blockDim.x + blockIdx.y % blockDim.x])
    return;
  int idx = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
  int mask = src_padding_mask[blockIdx.x * blockDim.x + threadIdx.x];
  float val = correlation[idx];

  float max_val = blockReduceMax(mask ? CUDA_FLOAT_INF_NEG : val);
  __shared__ float smax;
  if (threadIdx.x == 0) smax = max_val;
  __syncthreads();

  val = mask ? 0.f : expf(fmaxf(logit_thresh_min, val - smax));
  float rsum = blockReduceSum(val);
  __shared__ float ssum;
  if (threadIdx.x == 0) ssum = rsum;
  __syncthreads();

  correlation[idx] = val / (ssum + epsilon);
}

__global__ void ker_correlation_softmax_decself(float* correlation) {
  /**
  @brief
  query-key correlation softmax for decoder self attention

  @thread
  gridDim.x = batch_size * beam_size * head_num
  blockDim.x = cur_step + 1

  @param
  correlation: [batch_size, beam_size, head_num, cur_step + 1]
  */
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float val = correlation[idx];
  
  float max_val = blockReduceMax(val);
  __shared__ float smax;
  if (threadIdx.x == 0) smax = max_val;
  __syncthreads();

  val = expf(fmaxf(logit_thresh_min, val - smax));
  float rsum = blockReduceSum(val);
  __shared__ float ssum;
  if (threadIdx.x == 0) ssum = rsum;
  __syncthreads();

  correlation[idx] = val / ssum;
  //correlation[idx] = ssum;
}

__global__ void ker_correlation_softmax_encdec(float* correlation,
                                               const int* src_padding_mask) {
  /**
  @brief
  query-key correlation softmax for encoder-decoder attention

  @thread
  gridDim.x = batch_size
  gridDim.y = head_num * beam_size
  blockDim.x = batch_seq_len

  @param
  correlation: [batch_size, head_num, beam_size, batch_seq_len]
  src_padding_mask: [batch_size, batch_seq_len]
  */
  int idx = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
  int mask = src_padding_mask[blockIdx.x * blockDim.x + threadIdx.x];
  float val = correlation[idx]; 
  
  float max_val = blockReduceMax(mask ? CUDA_FLOAT_INF_NEG : val);
  __shared__ float smax;
  if (threadIdx.x == 0) smax = max_val;
  __syncthreads();

  val = mask ? 0.f : expf(fmaxf(logit_thresh_min, val - smax));
  float rsum = blockReduceSum(val);
  __shared__ float ssum;
  if (threadIdx.x == 0) ssum = rsum;
  __syncthreads();

  correlation[idx] = val / (ssum + epsilon);
}

__global__ void ker_arrange_atten_output(const float* ori_q, float* new_q,
                                         int beam_size, int dim_per_head,
                                         int head_num) {
  /**
  @brief
  reshape Scaled Dot-Product Attention output

  @thread
  gridDim.x = batch_token_num
  blockDim.x = hidden_size

  @param
  ori_q: [batch_size, head_num, beam_size, dim_per_head]
  new_q: [batch_size, beam_size, hidden_size]
  beam_size : for decoder, beam_size is beam_size; for encoder, beam_size is
  batch_seq_len dim_per_head: the head dim in Scaled Dot-Product Attention
  head_num: the head number in Scaled Dot-Product Attention
  token_emb: [hidden_size, vocab_size], note, it is different with encoder
  */

  int batch_id = blockIdx.x / beam_size;
  // note, for encoder, beam_id is token_id; for decoder, beam_id is beam_id
  int beam_id = blockIdx.x % beam_size;
  int head_id = threadIdx.x / dim_per_head;
  int dim_id = threadIdx.x % dim_per_head;
  new_q[blockIdx.x * blockDim.x + threadIdx.x] = ori_q[targetid_4dim(
      batch_id, head_id, beam_id, dim_id, head_num, beam_size, dim_per_head)];
}

__global__ void ker_refresh_result(
    const int* can_idx, const float* can_score, const int* num_can_per_beam,
    const int* old_alive_seq, int* new_alive_seq, float* seq_probs,
    float* seq_score, int* num_finish_beam, int vocab_size, int cur_step,
    float length_norm) {
  /**
  @brief
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

  finished_scores: [batch_size]
  cur_finished_scores: [batch_size, beam_size]
  finished_seq: [batch_size, max_step]
  */

  // step1 update alive_seq
  int can_pos = num_can_per_beam[blockIdx.x * gridDim.y] + blockIdx.y;  
  int ori_can_idx = can_idx[can_pos];  // can_beam_id * vocab_size + vocab_id
  int can_beam_id = ori_can_idx / vocab_size;
  int can_vocab_id = ori_can_idx % vocab_size;
  int thread_vocab_id;
  if (threadIdx.x > cur_step + 1) {
    thread_vocab_id = vocab_size - 1;
  } else if (threadIdx.x == cur_step + 1) {
    // add current step generate vocabulary id
    thread_vocab_id = can_vocab_id;
  } else {
    // threadIdx.x <= cur_step
    thread_vocab_id = old_alive_seq[targetid_3dim(blockIdx.x, can_beam_id, threadIdx.x, gridDim.y,
		    blockDim.x)];
  }
  new_alive_seq[targetid_3dim(blockIdx.x, blockIdx.y, threadIdx.x, gridDim.y,
                              blockDim.x)] = thread_vocab_id;

  // step2 update seq_probs if alive seq
  int eos_id = vocab_size - 1;
  if (can_vocab_id != eos_id) {
    // alive seq
    if (threadIdx.x == 0) {
      seq_probs[blockIdx.x * gridDim.y + blockIdx.y] =
          (can_score[can_pos] - blockIdx.x * min_log_probability) / length_norm;  // recover it
    }
    return;
  }
  
  // step3 update seq_score, num_finish_beam if finish seq
  if (threadIdx.x == 0) {
    atomicAdd(num_finish_beam, 1);
  }
  int seq_last_id = old_alive_seq[targetid_3dim(blockIdx.x, can_beam_id, cur_step, gridDim.y, blockDim.x)];
  // update finished seq score
  if (threadIdx.x == 0) {
    // note, with batch offset value, to sort between batch element
    seq_score[blockIdx.x * gridDim.y + blockIdx.y] = can_score[can_pos]; 
  }
}

__global__ void ker_refresh_cache(const int* num_can_per_beam,
                                  const int* can_idx, float* self_k_bgeem,
                                  float* self_v_bgeem, float* new_self_k_bgeem,
                                  float* new_self_v_bgeem,
                                  int self_k_bgeem_offset, int beam_size,
                                  int dim_per_head, int head_num,
                                  int vocab_size, int cur_step, int max_step) {
  /**
  @brief
  refresh K, V of self attention, add current step's projected k,v

  @thread
  gridDim.x = decoder_layer_num * (step_id + 1)
  gridDim.y = batch_size * beam_size * 2
  blockDim.x = hidden_size

  @param
  num_can_per_beam: [batch_size, beam_size]
  can_idx: [none], no certain length, determined by rough candidate number
  self_step_qkv: [batch_size, beam_size, 3, hidden_size] * decoder_layer_num
  self_k_bgeem: [batch_size, beam_size, head_num, dim_per_head, max_step] *
  decoder_layer_num self_v_bgeem: [batch_size, beam_size, head_num, max_step,
  dim_per_head] * decoder_layer_num


  self_step_qkv_offset = max_batch_size * beam_size * tw._hidden_size * 3
  self_k_bgeem_offset = max_batch_size * max_step * hidden_size * beam_size
  */
  int layer_id = blockIdx.x / (cur_step + 1);
  int step_id = blockIdx.x % (cur_step + 1);
  int kv_id = blockIdx.y & 1;
  int beam_id_global = blockIdx.y >> 1;
  int batch_id = beam_id_global / beam_size;
  int beam_id = beam_id_global % beam_size;
  int head_id = threadIdx.x / dim_per_head;
  int dim_id = threadIdx.x % dim_per_head;

  int can_pos = num_can_per_beam[batch_id * beam_size] + beam_id;
  int can_beam_id =
      can_idx[can_pos] / vocab_size;  // can_beam_id * vocab_size + vocab_id
  if (can_idx[can_pos] % vocab_size == vocab_size - 1) {
    return;
  }

  if (kv_id == 0) {
    // for key
    int base_pos = targetid_5dim(batch_id, 0, head_id, dim_id, step_id,
                                 beam_size, head_num, dim_per_head, max_step) +
                   layer_id * self_k_bgeem_offset;
    int beam_offset = blockDim.x * max_step;
    new_self_k_bgeem[base_pos + beam_offset * beam_id] =
        self_k_bgeem[base_pos + beam_offset * can_beam_id];
  } else {
    // for value
    int base_pos = targetid_5dim(batch_id, 0, head_id, step_id, dim_id,
                                 beam_size, head_num, max_step, dim_per_head) +
                   layer_id * self_k_bgeem_offset;
    int beam_offset = blockDim.x * max_step;
    new_self_v_bgeem[base_pos + beam_offset * beam_id] =
        self_v_bgeem[base_pos + beam_offset * can_beam_id];
  }
}

__global__ void ker_write_trg_tokenid_pos_penalty(const int* alive_seq,
		int* output, int max_step, int beam_size) {  
  /**
  @brief
  write result from alive seq to output, for length_penlty >= 0
  or length_penlty < 0 and decode to max_decode_step
  simply output the beam0 as final result

  @thread
  gridDim.x = batch_size
  blockDim.x = cur_step + 1

  @param
  alive_seq: [batch_size, beam_size, max_step], <start> is the first token in each beam
  output: [batch_size, cur_step + 1], no <start> and at least one <eos> in the last of seq
  */
  int target_id = targetid_3dim(blockIdx.x, 0, threadIdx.x + 1, beam_size, max_step);
  output[blockIdx.x * blockDim.x + threadIdx.x] = alive_seq[target_id];
}

__global__ void ker_write_trg_tokenid_neg_penalty(const int* alive_seq, const float* seq_score,
		int* output, int max_step, int beam_size, int vocab_size) {  
  /**
  @brief
  write result from alive seq to output, 
  for length_penlty < 0 and all beam has reach it's eos
  compute each beam's score and select the top beam

  @thread
  gridDim.x = batch_size
  blockDim.x = cur_step + 1

  @param
  alive_seq: [batch_size, beam_size, max_step], <start> is the first token in each beam
  seq_score: [batch_size, beam_size], the length_penlty < 0, seq_score is also the sum_log_probs
  output: [batch_size, cur_step + 1], no <start> and at least one <eos> in the last of seq
  */
  __shared__ float seq_final_score;
  __shared__ int res_beam_id;
  if (threadIdx.x == 0) {
    seq_final_score = CUDA_FLOAT_INF_NEG;
    res_beam_id = 0;
  }
  for(int beam_id=0; beam_id < beam_size; beam_id++) {
    int target_id = targetid_3dim(blockIdx.x, beam_id, threadIdx.x + 1, beam_size, max_step);
    int seq_len = blockReduceSum(int(alive_seq[target_id] != vocab_size - 1)); // compute seq len
    if (threadIdx.x == 0) {
      float cur_beam_score = seq_score[blockIdx.x * beam_size + beam_id] - blockIdx.x * min_log_probability;  // recover prob
      cur_beam_score /= (float(seq_len) + epsilon);
      if (cur_beam_score > seq_final_score) {
        seq_final_score = cur_beam_score;
	res_beam_id = beam_id;
      }
    }
    __syncthreads();
  }
  int target_id = targetid_3dim(blockIdx.x, res_beam_id, threadIdx.x + 1, beam_size, max_step);
  output[blockIdx.x * blockDim.x + threadIdx.x] = alive_seq[target_id];
  //output[blockIdx.x * blockDim.x + threadIdx.x] = int(seq_final_score[threadIdx.x]);
}

}  // namespace nmt
}  // namespace lab
