#include <random>

#include "common.h"
#include "XLMRKernels.h"
/**
@file
Implemented the cuda kernel function and its launcher
that required by multilingual nmt model.
Currently, fp16 and fp32 versions are provided
*/
namespace lightseq {
namespace cuda {
/**
@brief: ker_multilg_enc_emb
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
__global__ void ker_xlmr_enc_emb(const T* token_emb, const T* pos_emb,
                                 const T* src_lang_emb, const int* token_id,
                                 const int* lang_id, T* output,
                                 int* padding_mask, int padding_id,
                                 const int hidden_size) {
  int target_pos = blockIdx.x * gridDim.y + blockIdx.y;
  int start = target_pos * hidden_size + threadIdx.x;
  int end = (target_pos + 1) * hidden_size;
  int tid = token_id[target_pos];
  int lid = lang_id[blockIdx.x];
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
                pos_emb[blockIdx.y * hidden_size + offset] +
                src_lang_emb[lid * hidden_size + offset];
  }
}

template <>
__global__ void ker_xlmr_enc_emb<__half>(
    const __half* token_emb, const __half* pos_emb, const __half* src_lang_emb,
    const int* token_id, const int* lang_id, __half* output, int* padding_mask,
    int padding_id, const int half_hidden_size) {
  int target_pos = blockIdx.x * gridDim.y + blockIdx.y;
  int start = target_pos * half_hidden_size + threadIdx.x;
  int end = (target_pos + 1) * half_hidden_size;
  int tid = token_id[target_pos];
  int lid = lang_id[blockIdx.x];
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
    float2 le = __half22float2(
        ((const half2*)src_lang_emb)[lid * half_hidden_size + offset]);
    te.x = te.x + pe.x + le.x;
    te.y = te.y + pe.y + le.y;

    output_h[i] = __float22half2_rn(te);
  }
}

template <typename T>
void ker_xlmr_enc_emb_launcher(int batch_size, int batch_seq_len,
                               int hidden_size, cudaStream_t stream,
                               const T* token_emb, const T* pos_emb,
                               const T* src_lang_emb, const int* token_id,
                               const int* lang_id, T* output, int* padding_mask,
                               int padding_id, int max_thread_per_block) {
  ker_xlmr_enc_emb<T>
      <<<dim3(batch_size, batch_seq_len), max_thread_per_block, 0, stream>>>(
          token_emb, pos_emb, src_lang_emb, token_id, lang_id, output,
          padding_mask, padding_id, hidden_size);
}

template <>
void ker_xlmr_enc_emb_launcher<__half>(
    int batch_size, int batch_seq_len, int hidden_size, cudaStream_t stream,
    const __half* token_emb, const __half* pos_emb, const __half* src_lang_emb,
    const int* token_id, const int* lang_id, __half* output, int* padding_mask,
    int padding_id, int max_thread_per_block) {
  ker_xlmr_enc_emb<__half>
      <<<dim3(batch_size, batch_seq_len), max_thread_per_block, 0, stream>>>(
          token_emb, pos_emb, src_lang_emb, token_id, lang_id, output,
          padding_mask, padding_id, hidden_size / 2);
}

template void ker_xlmr_enc_emb_launcher<float>(
    int batch_size, int batch_seq_len, int hidden_size, cudaStream_t stream,
    const float* token_emb, const float* pos_emb, const float* src_lang_emb,
    const int* token_id, const int* lang_id, float* output, int* padding_mask,
    int padding_id, int max_thread_per_block);

template void ker_xlmr_enc_emb_launcher<__half>(
    int batch_size, int batch_seq_len, int hidden_size, cudaStream_t stream,
    const __half* token_emb, const __half* pos_emb, const __half* src_lang_emb,
    const int* token_id, const int* lang_id, __half* output, int* padding_mask,
    int padding_id, int max_thread_per_block);

/**
@brief: ker_multilg_dec_emb
for multilingual decoder, look up token embedding, add position embedding
and lang embedding

@thread
gridDim.x = batch_size * beam_size
blockDim.x = max_thread_per_block

@param
token_emb: [hidden_size, vocab_size], note, it is different with encoder
pos_emb: [max_step, hidden_size]
src_lang_emb: [lang_num, hidden_size]
trg_lang_emb: [lang_num, hidden_size]
token_id: input token id, [batch_size, beam_size, max_step]
output: result, [batch_size, beam_size, hidden_size]
step: current step
max_step: max decoder steps
vocab_size: vocabulary size
*/
template <typename T>
__global__ void ker_xlmr_dec_emb(const T* token_emb, const T* pos_emb,
                                 const T* trg_lang_emb, const int* token_id,
                                 const int* lang_id, T* output, int step,
                                 int max_step, int vocab_size, int hidden_size,
                                 int beam_size) {
  int batch_id = blockIdx.x / beam_size;
  int trg_lang_id = lang_id[batch_id];
  int token_idx = token_id[blockIdx.x * max_step + step];
  for (uint offset = threadIdx.x; offset < hidden_size; offset += blockDim.x) {
    output[blockIdx.x * hidden_size + offset] =
        token_emb[offset * vocab_size + token_idx] +
        pos_emb[step * hidden_size + offset] +
        trg_lang_emb[trg_lang_id * hidden_size + offset];
  }
}

template <typename T>
void ker_xlmr_dec_emb_launcher(int step_token_num, int hidden_size,
                               cudaStream_t stream, const T* token_emb,
                               const T* pos_emb, const T* trg_lang_emb,
                               const int* token_id, const int* lang_id,
                               T* output, int step, int max_step,
                               int vocab_size, int beam_size,
                               int max_thread_per_block) {
  ker_xlmr_dec_emb<T><<<step_token_num, max_thread_per_block, 0, stream>>>(
      token_emb, pos_emb, trg_lang_emb, token_id, lang_id, output, step,
      max_step, vocab_size, hidden_size, beam_size);
}

template void ker_xlmr_dec_emb_launcher<float>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const float* token_emb, const float* pos_emb, const float* trg_lang_emb,
    const int* token_id, const int* lang_id, float* output, int step,
    int max_step, int vocab_size, int beam_size, int max_thread_per_block);

template void ker_xlmr_dec_emb_launcher<__half>(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const __half* token_emb, const __half* pos_emb, const __half* trg_lang_emb,
    const int* token_id, const int* lang_id, __half* output, int step,
    int max_step, int vocab_size, int beam_size, int max_thread_per_block);

}  // namespace cuda
}  // namespace lightseq
