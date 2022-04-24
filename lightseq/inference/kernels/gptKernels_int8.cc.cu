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

/**
@brief: ker_gpt_embedding_int8
for encoder, look up token embedding, add position embedding

@thread
gridDim.x = batch_size
gridDim.y = token_seq_len
blockDim.x = hidden_size

@param
token_emb: [vocab_size, hidden_size]
pos_emb: [max_step, hidden_size]
token_id: input token id, [batch_size, token_seq_len]
output: result, [batch_size, token_seq_len, hidden_size]
real_seq_len: record seq len exclude padding, [batch_size]
padding_id, the padding_id, default 0
pos_offset: get real pos when decoding which gridDim.y=1
*/
template <typename T>
__global__ void ker_gpt_embedding_int8(const int8_t* token_emb, const T* pos_emb,
                                  const int* token_id, T* output,
                                  int* real_seq_len, int padding_id,
                                  int pos_offset, float dequant_scale) {
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
__global__ void ker_gpt_embedding_int8<__half>(const int8_t* token_emb,
                                          const __half* pos_emb,
                                          const int* token_id, __half* output,
                                          int* real_seq_len, int padding_id,
                                          int pos_offset, float dequant_scale) {
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
  te.x = float(cte.x) + pe.x;
  te.y = float(cte.y) + pe.y;
  output_h[target_pos * blockDim.x + threadIdx.x] = __float22half2_rn(te);
}

template <typename T>
void ker_gpt_embedding_int8_launcher(int batch_size, int batch_seq_len,
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
void ker_gpt_embedding_int8_launcher<__half>(int batch_size, int batch_seq_len,
                                        int hidden_size, cudaStream_t stream,
                                        const int8_t* token_emb,
                                        const __half* pos_emb,
                                        const int* token_id, __half* output,
                                        int* real_seq_len, int padding_id,
                                        int pos_offset, float dequant_scale) {
  ker_gpt_embedding_int8<__half>
      <<<dim3(batch_size, batch_seq_len), hidden_size / 2, 0, stream>>>(
          token_emb, pos_emb, token_id, output, real_seq_len, padding_id,
          pos_offset, dequant_scale);
}

template void ker_gpt_embedding_int8_launcher<float>(
    int batch_size, int batch_seq_len, int hidden_size, cudaStream_t stream,
    const int8_t* token_emb, const float* pos_emb, const int* token_id,
    float* output, int* real_seq_len, int padding_id, int pos_offset,
    float dequant_scale);

template void ker_gpt_embedding_int8_launcher<__half>(
    int batch_size, int batch_seq_len, int hidden_size, cudaStream_t stream,
    const int8_t* token_emb, const __half* pos_emb, const int* token_id,
    __half* output, int* real_seq_len, int padding_id, int pos_offset,
    float dequant_scale);

}  // namespace cuda
}  // namespace lightseq
