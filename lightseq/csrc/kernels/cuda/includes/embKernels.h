#pragma once
#include <cuda.h>
#include <cuda_fp16.h>

namespace lightseq {
namespace cuda {

void launch_split_multilg_request(const int *req, int *src_lang_id,
                                  int *trg_lang_id, int *src_token_id,
                                  int batch_size, int req_len,
                                  cudaStream_t &stream);

template <typename T>
void launch_enc_emb(const T *token_emb, const T *pos_emb, const int *tokens,
                    T *output, T *pad_mask, int pad_id, int batch_size,
                    int seq_len, int hidden_dim, cudaStream_t stream,
                    const T *lang_emb, const int *lang_id, int multilg_type);

template <typename T>
void launch_dec_emb(const T *token_emb, const T *pos_emb, int *tokens,
                    const T *lang_emb, const int *lang_id, T *output,
                    int batch_size, int beam_size, int hidden_dim,
                    int vocab_size, int step, int max_step, int multilg_type,
                    cudaStream_t stream);

template <typename T>
void launch_patch_emb(const T *conv_weight, const T *conv_bias,
                      const T *pos_emb, const T *cls_emb, const float *input,
                      T *output, int patch_size, int image_size, int batch_size,
                      int max_step, int hidden_dim, int channel_input,
                      cudaStream_t stream);

}  // namespace cuda
}  // namespace lightseq
