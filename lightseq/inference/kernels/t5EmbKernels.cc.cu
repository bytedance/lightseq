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
    value.x = 0.f;
    value.y = 0.f;
    value.z = 0.f;
    value.w = 0.f;
  } else {
    if (dim_idx == 0) {
      pad_mask[tokens_idx] = 0;
    }
    value = ((float4 *)token_emb)[token * hidden_dim + dim_idx];
    // float4 pemb = ((float4 *)pos_emb)[seq_idx * hidden_dim + dim_idx];
    // value.x += pemb.x;
    // value.y += pemb.y;
    // value.z += pemb.z;
    // value.w += pemb.w;
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
    value.x = 0.f;
    value.y = 0.f;
    value.z = 0.f;
    value.w = 0.f;
  } else {
    if (dim_idx == 0) {
      pad_mask[tokens_idx] = 0;
    }
    value = ((float4 *)token_emb)[token * hidden_dim + dim_idx];
    // float4 pemb = ((float4 *)pos_emb)[seq_idx * hidden_dim + dim_idx];
    __half2 *value_h2 = (__half2 *)(&value);
    // __half2 *pemb_h2 = (__half2 *)(&pemb);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 value_f2 = __half22float2(value_h2[i]);
      // float2 pemb_f2 = __half22float2(pemb_h2[i]);
      // value_f2.x += pemb_f2.x;
      // value_f2.y += pemb_f2.y;
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


}  // namespace cuda
}  // namespace lightseq
