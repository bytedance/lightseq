#include "common.h"
#include "embKernels.h"

/**
@file
Implemented the cuda kernel function and its launcher
that required by embedding layer in transformer model.
Currently, fp16 and fp32 versions are provided
*/
namespace lightseq {
namespace cuda {
/**
@brief: ker_split_multilg_request
the format of request in multilingual:
  e.g. <en> <de> <hello> <world> <.>
  request shape: [batch_size, src_seq_len + 2]
  request = numpy.concatenate((src_lang_id, trg_lang_id, src_token_id), axis=1)

@thread
gridDim.x = (nele + MAX_THREADS - 1) / MAX_THREADS
blockDim.x = MAX_THREADS

@param
req: [batch_size, src_seq_len + 2, hidden_dim]
src_lang_id: [batch_size]
trg_lang_id: [batch_size]
src_token_id: [batch_size, src_seq_len, hidden_dim]
req_len: src_seq_len + 2
*/
__global__ void ker_split_multilg_request(const int *req, int *src_lang_id,
                                          int *trg_lang_id, int *src_token_id,
                                          int batch_size, int req_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < batch_size * req_len) {
    int value = req[idx];
    int seq_id = idx / req_len;
    int token_id = idx % req_len;

    if (token_id == 0) {
      src_lang_id[seq_id] = value;
    } else if (token_id == 1) {
      trg_lang_id[seq_id] = value;
    } else {
      int new_idx = flat_2dim(seq_id, token_id - 2, req_len - 2);
      src_token_id[new_idx] = value;
    }
  }
}

void launch_split_multilg_request(const int *req, int *src_lang_id,
                                  int *trg_lang_id, int *src_token_id,
                                  int batch_size, int req_len,
                                  cudaStream_t &stream) {
  if (req_len < 3) {
    throw std::runtime_error("req_len should be greater than 2");
  }
  int nele = batch_size * req_len;
  int nblock = (nele + MAX_THREADS - 1) / MAX_THREADS;
  ker_split_multilg_request<<<nblock, MAX_THREADS, 0, stream>>>(
      req, src_lang_id, trg_lang_id, src_token_id, batch_size, req_len);
}

/**
@brief: ker_enc_emb
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
__global__ void ker_enc_emb(const T *token_emb, const T *pos_emb,
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
    float4 pemb = ((float4 *)pos_emb)[seq_idx * hidden_dim + dim_idx];
    value.x += pemb.x;
    value.y += pemb.y;
    value.z += pemb.z;
    value.w += pemb.w;
  }
  ((float4 *)output)[idx] = value;
}

template <>
__global__ void ker_enc_emb<__half>(const __half *token_emb,
                                    const __half *pos_emb, const int *tokens,
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
    float4 pemb = ((float4 *)pos_emb)[seq_idx * hidden_dim + dim_idx];
    __half2 *value_h2 = (__half2 *)(&value);
    __half2 *pemb_h2 = (__half2 *)(&pemb);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 value_f2 = __half22float2(value_h2[i]);
      float2 pemb_f2 = __half22float2(pemb_h2[i]);
      value_f2.x += pemb_f2.x;
      value_f2.y += pemb_f2.y;
      value_h2[i] = __float22half2_rn(value_f2);
    }
  }
  ((float4 *)output)[idx] = value;
}

/**
@brief: ker_enc_emb_multilg_token
for encoder, look up token embedding, add position embedding

@thread
gridDim.x = (nele + MAX_THREADS - 1) / MAX_THREADS
blockDim.x = MAX_THREADS;

@param
token_emb: [vocab_size, hidden_dim]
pos_emb: [max_step, hidden_dim]
tokens: input token id, [batch_size, seq_len]
lang_emb: language embedding, [num_lang, hidden_dim]
lang_id: language index, [batch_size]
output: result, [batch_size, seq_len, hidden_dim]
pad_mask: record the padding token, [batch_size, seq_len]
pad_id, the padding token id
*/
template <typename T>
__global__ void ker_enc_emb_multilg_token(const T *token_emb, const T *pos_emb,
                                          const int *tokens, const T *lang_emb,
                                          const int *lang_id, T *output,
                                          int *pad_mask, int pad_id,
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

    // add pos emb
    float4 pemb = ((float4 *)pos_emb)[seq_idx * hidden_dim + dim_idx];
    value.x += pemb.x;
    value.y += pemb.y;
    value.z += pemb.z;
    value.w += pemb.w;
    // add lang emb
    pemb = ((float4 *)lang_emb)[lang_id[batch_idx] * hidden_dim + dim_idx];
    value.x += pemb.x;
    value.y += pemb.y;
    value.z += pemb.z;
    value.w += pemb.w;
  }
  ((float4 *)output)[idx] = value;
}

template <>
__global__ void ker_enc_emb_multilg_token<__half>(
    const __half *token_emb, const __half *pos_emb, const int *tokens,
    const __half *lang_emb, const int *lang_id, __half *output, int *pad_mask,
    int pad_id, int batch_size, int seq_len, int hidden_dim) {
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
    __half2 *value_h2 = (__half2 *)(&value);

    float4 pemb = ((float4 *)pos_emb)[seq_idx * hidden_dim + dim_idx];
    __half2 *pemb_h2 = (__half2 *)(&pemb);
    float4 lemb =
        ((float4 *)lang_emb)[lang_id[batch_idx] * hidden_dim + dim_idx];
    __half2 *lemb_h2 = (__half2 *)(&lemb);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 value_f2 = __half22float2(value_h2[i]);
      float2 pemb_f2 = __half22float2(pemb_h2[i]);
      float2 lemb_f2 = __half22float2(lemb_h2[i]);
      value_f2.x += pemb_f2.x + lemb_f2.x;
      value_f2.y += pemb_f2.y + lemb_f2.y;
      value_h2[i] = __float22half2_rn(value_f2);
    }
  }
  ((float4 *)output)[idx] = value;
}

/**
@brief: ker_enc_emb_multilg_sentence
for encoder, look up token embedding, add position embedding

@thread
gridDim.x = (nele + MAX_THREADS - 1) / MAX_THREADS
blockDim.x = MAX_THREADS;

@param
token_emb: [vocab_size, hidden_dim]
pos_emb: [max_step, hidden_dim]
tokens: input token id, [batch_size, seq_len]
lang_emb: language embedding, [num_lang, hidden_dim]
lang_id: language index, [batch_size]
output: result, [batch_size, seq_len, hidden_dim]
pad_mask: record the padding token, [batch_size, seq_len]
pad_id, the padding token id
*/
template <typename T>
__global__ void ker_enc_emb_multilg_sentence(
    const T *token_emb, const T *pos_emb, const int *tokens, const T *lang_emb,
    const int *lang_id, T *output, int *pad_mask, int pad_id, int batch_size,
    int seq_len, int hidden_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size * seq_len * hidden_dim) {
    return;
  }
  int batch_idx, seq_idx, dim_idx;
  decompose_3dim(idx, seq_len, hidden_dim, &batch_idx, &seq_idx, &dim_idx);

  bool is_pad;
  int token_emb_idx;
  if (seq_idx == 0) {
    is_pad = false;
    token_emb = lang_emb;
    token_emb_idx = lang_id[batch_idx];
  } else {
    token_emb_idx = tokens[batch_idx * (seq_len - 1) + seq_idx - 1];
    is_pad = (token_emb_idx == pad_id);
  }

  float4 value;
  int tokens_idx = batch_idx * seq_len + seq_idx;
  if (is_pad) {
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
    value = ((float4 *)token_emb)[token_emb_idx * hidden_dim + dim_idx];
    float4 pemb = ((float4 *)pos_emb)[seq_idx * hidden_dim + dim_idx];
    value.x += pemb.x;
    value.y += pemb.y;
    value.z += pemb.z;
    value.w += pemb.w;
  }
  ((float4 *)output)[idx] = value;
}

template <>
__global__ void ker_enc_emb_multilg_sentence<__half>(
    const __half *token_emb, const __half *pos_emb, const int *tokens,
    const __half *lang_emb, const int *lang_id, __half *output, int *pad_mask,
    int pad_id, int batch_size, int seq_len, int hidden_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size * seq_len * hidden_dim) {
    return;
  }
  int batch_idx, seq_idx, dim_idx;
  decompose_3dim(idx, seq_len, hidden_dim, &batch_idx, &seq_idx, &dim_idx);

  bool is_pad;
  int token_emb_idx;
  if (seq_idx == 0) {
    is_pad = false;
    token_emb = lang_emb;
    token_emb_idx = lang_id[batch_idx];
  } else {
    token_emb_idx = tokens[batch_idx * (seq_len - 1) + seq_idx - 1];
    is_pad = (token_emb_idx == pad_id);
  }

  float4 value;
  int tokens_idx = batch_idx * seq_len + seq_idx;
  if (is_pad) {
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
    value = ((float4 *)token_emb)[token_emb_idx * hidden_dim + dim_idx];
    float4 pemb = ((float4 *)pos_emb)[seq_idx * hidden_dim + dim_idx];
    __half2 *value_h2 = (__half2 *)(&value);
    __half2 *pemb_h2 = (__half2 *)(&pemb);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 value_f2 = __half22float2(value_h2[i]);
      float2 pemb_f2 = __half22float2(pemb_h2[i]);
      value_f2.x += pemb_f2.x;
      value_f2.y += pemb_f2.y;
      value_h2[i] = __float22half2_rn(value_f2);
    }
  }
  ((float4 *)output)[idx] = value;
}

template <typename T>
void launch_enc_emb(const T *token_emb, const T *pos_emb, const int *tokens,
                    T *output, int *pad_mask, int pad_id, int batch_size,
                    int seq_len, int hidden_dim, cudaStream_t stream,
                    const T *lang_emb, const int *lang_id, int multilg_type) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  hidden_dim >>= 2;
  int nele = batch_size * seq_len * hidden_dim;
  int nblock = (nele + MAX_THREADS - 1) / MAX_THREADS;
  if (multilg_type == 0) {
    ker_enc_emb<T><<<nblock, MAX_THREADS, 0, stream>>>(
        token_emb, pos_emb, tokens, output, pad_mask, pad_id, batch_size,
        seq_len, hidden_dim);
  } else if (multilg_type == 1) {
    ker_enc_emb_multilg_token<T><<<nblock, MAX_THREADS, 0, stream>>>(
        token_emb, pos_emb, tokens, lang_emb, lang_id, output, pad_mask, pad_id,
        batch_size, seq_len, hidden_dim);
  } else {
    ker_enc_emb_multilg_sentence<T><<<nblock, MAX_THREADS, 0, stream>>>(
        token_emb, pos_emb, tokens, lang_emb, lang_id, output, pad_mask, pad_id,
        batch_size, seq_len, hidden_dim);
  }
}

template <>
void launch_enc_emb<__half>(const __half *token_emb, const __half *pos_emb,
                            const int *tokens, __half *output, int *pad_mask,
                            int pad_id, int batch_size, int seq_len,
                            int hidden_dim, cudaStream_t stream,
                            const __half *lang_emb, const int *lang_id,
                            int multilg_type) {
  if (hidden_dim % 8 != 0) {
    throw std::runtime_error("violate hidden_dim % 8 = 0");
  }
  hidden_dim >>= 3;
  int nele = batch_size * seq_len * hidden_dim;
  int nblock = (nele + MAX_THREADS - 1) / MAX_THREADS;

  if (multilg_type == 0) {
    ker_enc_emb<__half><<<nblock, MAX_THREADS, 0, stream>>>(
        token_emb, pos_emb, tokens, output, pad_mask, pad_id, batch_size,
        seq_len, hidden_dim);
  } else if (multilg_type == 1) {
    ker_enc_emb_multilg_token<__half><<<nblock, MAX_THREADS, 0, stream>>>(
        token_emb, pos_emb, tokens, lang_emb, lang_id, output, pad_mask, pad_id,
        batch_size, seq_len, hidden_dim);
  } else {
    ker_enc_emb_multilg_sentence<__half><<<nblock, MAX_THREADS, 0, stream>>>(
        token_emb, pos_emb, tokens, lang_emb, lang_id, output, pad_mask, pad_id,
        batch_size, seq_len, hidden_dim);
  }
}

template void launch_enc_emb<float>(const float *token_emb,
                                    const float *pos_emb, const int *tokens,
                                    float *output, int *pad_mask, int pad_id,
                                    int batch_size, int seq_len, int hidden_dim,
                                    cudaStream_t stream, const float *lang_emb,
                                    const int *lang_id, int multilg_type);

template void launch_enc_emb<__half>(const __half *token_emb,
                                     const __half *pos_emb, const int *tokens,
                                     __half *output, int *pad_mask, int pad_id,
                                     int batch_size, int seq_len,
                                     int hidden_dim, cudaStream_t stream,
                                     const __half *lang_emb, const int *lang_id,
                                     int multilg_type);

/**
@brief: ker_dec_embedding
for decoder, look up token embedding, add position embedding

@thread
gridDim.x = (nele + MAX_THREADS - 1) / MAX_THREADS;
blockDim.x = MAX_THREADS

@param
token_emb: [hidden_dim, vocab_size], note, it is different with encoder
pos_emb: [max_step, hidden_dim]
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
__global__ void ker_dec_emb(const T *token_emb, const T *pos_emb, int *tokens,
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
  if (multilg_type == 2 && step == 0) {
    // the bos of sentense level multilg is target lang id
    int lid = lang_id[batch_idx];
    emb = lang_emb[flat_2dim(lid, dim_idx, hidden_dim)];
    tokens[flat_3dim(batch_idx, beam_idx, 0, beam_size, max_step)] = lid;
  } else {
    int token =
        tokens[flat_3dim(batch_idx, beam_idx, step, beam_size, max_step)];
    emb = token_emb[flat_2dim(dim_idx, token, vocab_size)];
  }
  float value =
      float(emb) + float(pos_emb[flat_2dim(step, dim_idx, hidden_dim)]);
  if (multilg_type == 1) {
    // token level multilg, add lang_emb
    value +=
        float(lang_emb[flat_2dim(lang_id[batch_idx], dim_idx, hidden_dim)]);
  }
  output[idx] = T(value);
}

template <typename T>
void launch_dec_emb(const T *token_emb, const T *pos_emb, int *tokens,
                    const T *lang_emb, const int *lang_id, T *output,
                    int batch_size, int beam_size, int hidden_dim,
                    int vocab_size, int step, int max_step, int multilg_type,
                    cudaStream_t stream) {
  if (step >= max_step) {
    throw std::runtime_error("violate step < max_step");
  }
  int nele = batch_size * beam_size * hidden_dim;
  int nblock = (nele + MAX_THREADS - 1) / MAX_THREADS;
  ker_dec_emb<T><<<nblock, MAX_THREADS, 0, stream>>>(
      token_emb, pos_emb, tokens, lang_emb, lang_id, output, batch_size,
      beam_size, hidden_dim, vocab_size, step, max_step, multilg_type);
}

template void launch_dec_emb<float>(const float *token_emb,
                                    const float *pos_emb, int *tokens,
                                    const float *lang_emb, const int *lang_id,
                                    float *output, int batch_size,
                                    int beam_size, int hidden_dim,
                                    int vocab_size, int step, int max_step,
                                    int multilg_type, cudaStream_t stream);

template void launch_dec_emb<__half>(const __half *token_emb,
                                     const __half *pos_emb, int *tokens,
                                     const __half *lang_emb, const int *lang_id,
                                     __half *output, int batch_size,
                                     int beam_size, int hidden_dim,
                                     int vocab_size, int step, int max_step,
                                     int multilg_type, cudaStream_t stream);
}  // namespace cuda
}  // namespace lightseq
