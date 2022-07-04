#include "common.h"
#include "t5EmbKernels.h"

/**
@file
Implemented the cuda kernel function and its launcher
that required by embedding layer in transformer model.
Currently, fp16 and fp32 versions are provided
*/
namespace lightseq {
namespace cuda {

/**
@brief: t5_ker_enc_emb
for encoder, look up token embedding, add position embedding

@thread
gridDim.x = (nele + MAX_THREADS - 1) / MAX_THREADS
blockDim.x = MAX_THREADS;

@param
token_emb: [vocab_size, hidden_dim]
pos_emb: [max_step, hidden_dim]
tokens: input token id, [batch_size, seq_len]
output: result, [batch_size, seq_len, hidden_dim]
pad_mask: record the padding token, [batch_size, seq_len]
pad_id, the padding token id
*/
template <typename T>
__global__ void t5_ker_enc_emb(const T *token_emb,
                            const int *tokens, T *output, int *pad_mask,
                            int pad_id, int batch_size, int seq_len,
                            int hidden_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size * seq_len * hidden_dim) {
    return;
  }
  int batch_idx, seq_idx, dim_idx;
  decompose_3dim(idx, seq_len, hidden_dim, &batch_idx, &seq_idx, &dim_idx);
  int tokens_idx = batch_idx * seq_len + seq_idx;
  int token = tokens[tokens_idx];
  float4 value;

  if (token == pad_id) {
    if (dim_idx == 0) {
      pad_mask[tokens_idx] = 1;
    }
  } else {
    if (dim_idx == 0) {
      pad_mask[tokens_idx] = 0;
    }
    value = ((float4 *)token_emb)[token * hidden_dim + dim_idx];
  }
  ((float4 *)output)[idx] = value;
}



template <>
__global__ void t5_ker_enc_emb<__half>(const __half *token_emb, const int *tokens,
                                    __half *output, int *pad_mask, int pad_id,
                                    int batch_size, int seq_len,
                                    int hidden_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size * seq_len * hidden_dim) {
    return;
  }
  int batch_idx, seq_idx, dim_idx;
  decompose_3dim(idx, seq_len, hidden_dim, &batch_idx, &seq_idx, &dim_idx);
  int tokens_idx = batch_idx * seq_len + seq_idx;
  int token = tokens[tokens_idx];
  float4 value;

  if (token == pad_id) {
    if (dim_idx == 0) {
      pad_mask[tokens_idx] = 1;
    }
  } else {
    if (dim_idx == 0) {
      pad_mask[tokens_idx] = 0;
    }
    value = ((float4 *)token_emb)[token * hidden_dim + dim_idx];
    __half2 *value_h2 = (__half2 *)(&value);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 value_f2 = __half22float2(value_h2[i]);
      value_h2[i] = __float22half2_rn(value_f2);
    }
  }
  ((float4 *)output)[idx] = value;
}



template <typename T>
void t5_launch_enc_emb(const T *token_emb, const int *tokens,
                    T *output, int *pad_mask, int pad_id, int batch_size,
                    int seq_len, int hidden_dim, cudaStream_t stream,
                    const T *lang_emb, const int *lang_id) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  hidden_dim >>= 2;
  int nele = batch_size * seq_len * hidden_dim;
  int nblock = (nele + MAX_THREADS - 1) / MAX_THREADS;
  
  t5_ker_enc_emb<T><<<nblock, MAX_THREADS, 0, stream>>>(
        token_emb, tokens, output, pad_mask, pad_id, batch_size,
        seq_len, hidden_dim);
}


template <>
void t5_launch_enc_emb<__half>(const __half *token_emb,
                            const int *tokens, __half *output, int *pad_mask,
                            int pad_id, int batch_size, int seq_len,
                            int hidden_dim, cudaStream_t stream,
                            const __half *lang_emb, const int *lang_id) {
  if (hidden_dim % 8 != 0) {
    throw std::runtime_error("violate hidden_dim % 8 = 0");
  }
  hidden_dim >>= 3;
  int nele = batch_size * seq_len * hidden_dim;
  int nblock = (nele + MAX_THREADS - 1) / MAX_THREADS;


  t5_ker_enc_emb<__half><<<nblock, MAX_THREADS, 0, stream>>>(
        token_emb, tokens, output, pad_mask, pad_id, batch_size,
        seq_len, hidden_dim);
}

template void t5_launch_enc_emb<float>(const float *token_emb, const int *tokens,
                                    float *output, int *pad_mask, int pad_id,
                                    int batch_size, int seq_len, int hidden_dim,
                                    cudaStream_t stream, const float *lang_emb,
                                    const int *lang_id);

template void t5_launch_enc_emb<__half>(const __half *token_emb, const int *tokens,
                                     __half *output, int *pad_mask, int pad_id,
                                     int batch_size, int seq_len,
                                     int hidden_dim, cudaStream_t stream,
                                     const __half *lang_emb, const int *lang_id);



/**
@brief: t5_ker_dec_embedding
for decoder, look up token embedding, add position embedding

@thread
gridDim.x = (nele + MAX_THREADS - 1) / MAX_THREADS;
blockDim.x = MAX_THREADS

@param
token_emb: [hidden_dim, vocab_size], note, it is different with encoder
tokens: input token id, [batch_size, beam_size, max_step]
lang_emb: language embedding, [num_lang, hidden_dim]
lang_id: language index, [batch_size]
output: result, [batch_size, beam_size, hidden_dim]
step: current decoder step
max_step: max decoder steps
multilg_type: 0 for no multilg, 1 for token level multilg,
  2 for sentence level multilg
*/
template <typename T>
__global__ void t5_ker_dec_emb(const T *token_emb, int *tokens,
                            const T *lang_emb, const int *lang_id, T *output,
                            int batch_size, int beam_size, int hidden_dim,
                            int vocab_size, int step, int max_step,
                            int multilg_type) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size * beam_size * hidden_dim) {
    return;
  }
  int batch_idx, beam_idx, dim_idx;
  decompose_3dim(idx, beam_size, hidden_dim, &batch_idx, &beam_idx, &dim_idx);

  T emb;
  int token =
      tokens[flat_3dim(batch_idx, beam_idx, step, beam_size, max_step)];
  emb = token_emb[flat_2dim(dim_idx, token, vocab_size)];

  float value = float(emb);
  output[idx] = T(value);
}


template <typename T>
void t5_launch_dec_emb(const T *token_emb, int *tokens,
                    const T *lang_emb, const int *lang_id, T *output,
                    int batch_size, int beam_size, int hidden_dim,
                    int vocab_size, int step, int max_step, int multilg_type,
                    cudaStream_t stream) {
  if (step >= max_step) {
    throw std::runtime_error("violate step < max_step");
  }
  int nele = batch_size * beam_size * hidden_dim;
  int nblock = (nele + MAX_THREADS - 1) / MAX_THREADS;
  t5_ker_dec_emb<T><<<nblock, MAX_THREADS, 0, stream>>>(
      token_emb, tokens, lang_emb, lang_id, output, batch_size,
      beam_size, hidden_dim, vocab_size, step, max_step, multilg_type);
}

template void t5_launch_dec_emb<float>(const float *token_emb, int *tokens,
                                    const float *lang_emb, const int *lang_id,
                                    float *output, int batch_size,
                                    int beam_size, int hidden_dim,
                                    int vocab_size, int step, int max_step,
                                    int multilg_type, cudaStream_t stream);

template void t5_launch_dec_emb<__half>(const __half *token_emb, int *tokens,
                                     const __half *lang_emb, const int *lang_id,
                                     __half *output, int batch_size,
                                     int beam_size, int hidden_dim,
                                     int vocab_size, int step, int max_step,
                                     int multilg_type, cudaStream_t stream);


}  // namespace cuda
}  // namespace lightseq
