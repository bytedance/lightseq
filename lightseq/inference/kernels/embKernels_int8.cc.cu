#include "common.h"
#include "embKernels_int8.h"

/**
@file
Implemented the cuda kernel function and its launcher
that required by embedding layer in transformer model.
Currently, fp16 and fp32 versions are provided
*/
namespace lightseq {
namespace cuda {

template <typename T>
__global__ void ker_enc_emb_i8I(const int8_t *token_emb, const T *pos_emb,
                                const int *tokens, T *output, int *pad_mask,
                                int pad_id, int batch_size, int seq_len,
                                int hidden_dim, float dequant_scale) {
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
    char4 value_i4 = ((char4 *)token_emb)[token * hidden_dim + dim_idx];
    float4 pemb = ((float4 *)pos_emb)[seq_idx * hidden_dim + dim_idx];
    float scale = dequant_scale * sqrtf(hidden_dim);
    value.x = float(value_i4.x) * scale + pemb.x;
    value.y = float(value_i4.y) * scale + pemb.y;
    value.z = float(value_i4.z) * scale + pemb.z;
    value.w = float(value_i4.w) * scale + pemb.w;
  }
  ((float4 *)output)[idx] = value;
}

template <>
__global__ void ker_enc_emb_i8I<__half>(const int8_t *token_emb,
                                        const __half *pos_emb,
                                        const int *tokens, __half *output,
                                        int *pad_mask, int pad_id,
                                        int batch_size, int seq_len,
                                        int hidden_dim, float dequant_scale) {
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
    int2 value_i8 = ((int2 *)token_emb)[token * hidden_dim + dim_idx];
    float4 pemb = ((float4 *)pos_emb)[seq_idx * hidden_dim + dim_idx];
    __half2 *value_h2 = (__half2 *)(&value);
    char2 *value_i2 = (char2 *)(&value_i8);
    __half2 *pemb_h2 = (__half2 *)(&pemb);
    float scale = dequant_scale * sqrtf(hidden_dim);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 value_f2;
      float2 pemb_f2 = __half22float2(pemb_h2[i]);
      value_f2.x = float(__half(float(value_i2[i].x) * scale)) + pemb_f2.x;
      value_f2.y = float(__half(float(value_i2[i].y) * scale)) + pemb_f2.y;
      value_h2[i] = __float22half2_rn(value_f2);
    }
  }
  ((float4 *)output)[idx] = value;
}

template <typename T>
void launch_enc_emb_i8I(const int8_t *token_emb, const T *pos_emb,
                        const int *tokens, T *output, int *pad_mask, int pad_id,
                        int batch_size, int seq_len, int hidden_dim,
                        cudaStream_t stream, const T *lang_emb,
                        const int *lang_id, int multilg_type,
                        float dequant_scale) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  hidden_dim >>= 2;
  int nele = batch_size * seq_len * hidden_dim;
  int nblock = (nele + MAX_THREADS - 1) / MAX_THREADS;
  if (multilg_type == 0) {
    ker_enc_emb_i8I<T><<<nblock, MAX_THREADS, 0, stream>>>(
        token_emb, pos_emb, tokens, output, pad_mask, pad_id, batch_size,
        seq_len, hidden_dim, dequant_scale);
  } else {
    throw std::runtime_error("multilingle not supported");
  }
}

template <>
void launch_enc_emb_i8I<__half>(const int8_t *token_emb, const __half *pos_emb,
                                const int *tokens, __half *output,
                                int *pad_mask, int pad_id, int batch_size,
                                int seq_len, int hidden_dim,
                                cudaStream_t stream, const __half *lang_emb,
                                const int *lang_id, int multilg_type,
                                float dequant_scale) {
  if (hidden_dim % 8 != 0) {
    throw std::runtime_error("violate hidden_dim % 8 = 0");
  }
  hidden_dim >>= 3;
  int nele = batch_size * seq_len * hidden_dim;
  int nblock = (nele + MAX_THREADS - 1) / MAX_THREADS;

  if (multilg_type == 0) {
    ker_enc_emb_i8I<__half><<<nblock, MAX_THREADS, 0, stream>>>(
        token_emb, pos_emb, tokens, output, pad_mask, pad_id, batch_size,
        seq_len, hidden_dim, dequant_scale);
  } else {
    throw std::runtime_error("multilingle not supported");
  }
}

template void launch_enc_emb_i8I<float>(
    const int8_t *token_emb, const float *pos_emb, const int *tokens,
    float *output, int *pad_mask, int pad_id, int batch_size, int seq_len,
    int hidden_dim, cudaStream_t stream, const float *lang_emb,
    const int *lang_id, int multilg_type, float dequant_scale);

template void launch_enc_emb_i8I<__half>(
    const int8_t *token_emb, const __half *pos_emb, const int *tokens,
    __half *output, int *pad_mask, int pad_id, int batch_size, int seq_len,
    int hidden_dim, cudaStream_t stream, const __half *lang_emb,
    const int *lang_id, int multilg_type, float dequant_scale);

template <typename T>
__global__ void ker_dec_emb_i8I(const int8_t *token_emb, const T *pos_emb,
                                int *tokens, const T *lang_emb,
                                const int *lang_id, T *output, int batch_size,
                                int beam_size, int hidden_dim, int vocab_size,
                                int step, int max_step, int multilg_type,
                                float dequant_scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size * beam_size * hidden_dim) {
    return;
  }
  int batch_idx, beam_idx, dim_idx;
  decompose_3dim(idx, beam_size, hidden_dim, &batch_idx, &beam_idx, &dim_idx);

  int8_t emb;
  int token = tokens[flat_3dim(batch_idx, beam_idx, step, beam_size, max_step)];
  emb = token_emb[flat_2dim(dim_idx, token, vocab_size)];
  float value = float(T(float(emb) * dequant_scale * sqrtf(hidden_dim))) +
                float(pos_emb[flat_2dim(step, dim_idx, hidden_dim)]);
  output[idx] = T(value);
}

template <typename T>
void launch_dec_emb_i8I(const int8_t *token_emb, const T *pos_emb, int *tokens,
                        const T *lang_emb, const int *lang_id, T *output,
                        int batch_size, int beam_size, int hidden_dim,
                        int vocab_size, int step, int max_step,
                        int multilg_type, cudaStream_t stream,
                        float dequant_scale) {
  if (step >= max_step) {
    throw std::runtime_error("violate step < max_step");
  }
  if (multilg_type != 0) {
    throw std::runtime_error("multilingle not supported");
  }
  int nele = batch_size * beam_size * hidden_dim;
  int nblock = (nele + MAX_THREADS - 1) / MAX_THREADS;
  ker_dec_emb_i8I<T><<<nblock, MAX_THREADS, 0, stream>>>(
      token_emb, pos_emb, tokens, lang_emb, lang_id, output, batch_size,
      beam_size, hidden_dim, vocab_size, step, max_step, multilg_type,
      dequant_scale);
}

template void launch_dec_emb_i8I<float>(
    const int8_t *token_emb, const float *pos_emb, int *tokens,
    const float *lang_emb, const int *lang_id, float *output, int batch_size,
    int beam_size, int hidden_dim, int vocab_size, int step, int max_step,
    int multilg_type, cudaStream_t stream, float dequant_scale);

template void launch_dec_emb_i8I<__half>(
    const int8_t *token_emb, const __half *pos_emb, int *tokens,
    const __half *lang_emb, const int *lang_id, __half *output, int batch_size,
    int beam_size, int hidden_dim, int vocab_size, int step, int max_step,
    int multilg_type, cudaStream_t stream, float dequant_scale);
}  // namespace cuda
}  // namespace lightseq
