#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>

#include "3rdparty/cub/cub/cub.cuh"

namespace lightseq {
namespace cuda {

const float logit_thresh_max = 64.f;
const float logit_thresh_min = -64.f;
const float min_log_probability = -2000.f;
const float epsilon = 0.000000000001;

template <typename T>
void ker_enc_embedding_launcher(int batch_size, int batch_seq_len,
                                int hidden_size, cudaStream_t stream,
                                const T* token_emb, const T* pos_emb,
                                const int* token_id, T* output,
                                int* padding_mask, int padding_id,
                                int max_thread_per_block);

template <typename T>
void ker_norm_layer_launcher(int token_num, int hidden_size,
                             cudaStream_t stream, T* matrix, const T* scale,
                             const T* bias, int max_thread_per_block);

template <typename T>
void ker_norm_layer_resual_launcher(int token_num, int hidden_size,
                                    cudaStream_t stream, T* input, T* output,
                                    const T* scale, const T* bias,
                                    const T* residual_bias,
                                    const int max_thread_per_block,
                                    bool is_post_ln = false);

template <typename T>
void select_beam_rough_topk_launcher(
    const T* logits, const T* logit_bias, const float* seq_probs,
    const float* seq_score, const int* alive_seq, int* can_idx,
    float* can_score, int* num_beam_can, int vocab_size, int max_step,
    float length_norm, int cur_step, int step_token_num,
    int max_thread_per_block, cudaStream_t stream, int beam_size,
    float diverse_lambda, int end_id);

void ker_diverse_beam_search_launcher(float* can_score, int* can_ids,
                                      int* num_beam_can, int step_token_num,
                                      int max_thread_per_block,
                                      cudaStream_t stream, int beam_size,
                                      float diverse_lambda, int vocab_size);

template <typename T>
void ker_bias_relu_launcher(int batch_token_num, int block_dim,
                            cudaStream_t stream, T* input, const T* bias,
                            int feature_dim);

template <typename T>
void ker_dec_embedding_launcher(int step_token_num, int hidden_size,
                                cudaStream_t stream, const T* token_emb,
                                const T* pos_emb, const int* token_id,
                                T* output, int step, int max_step,
                                int vocab_size, int max_thread_per_block);

template <typename T>
void ker_arrange_encself_qkv_launcher(int batch_token_num, int hidden_size,
                                      cudaStream_t stream, const T* ori_qkv,
                                      const T* qkv_bias, T* new_qkv,
                                      int max_batch_dim, int batch_seq_len,
                                      int dim_per_head, int head_num,
                                      int max_thread_per_block);

template <typename T>
void ker_arrange_decself_qkv_launcher(int step_token_num, int hidden_size,
                                      cudaStream_t stream, const T* ori_qkv,
                                      const T* qkv_bias, T* new_q, T* new_k,
                                      T* new_v, int head_num, int dim_per_head,
                                      int max_step, int step_id,
                                      int max_thread_per_block);

template <typename T>
void ker_refresh_cache_launcher(
    int grid_dim_x, int grid_dim_y, int block_dim, cudaStream_t stream,
    const int* num_can_per_beam, const int* can_idx, const T* self_k_bgeem,
    const T* self_v_bgeem, T* new_self_k_bgeem, T* new_self_v_bgeem,
    int self_k_bgeem_offset, int beam_size, int dim_per_head, int head_num,
    int vocab_size, int cur_step, int max_step, bool diverse, int end_id);

template <typename T>
void ker_arrange_encdec_kv_launcher(int batch_token_num, int dec_layer_num,
                                    int hidden_size, cudaStream_t stream,
                                    const T* ori_kv, const T* kv_bias, T* new_k,
                                    T* new_v, int offset_per_layer,
                                    int batch_seq_len, int dim_per_head,
                                    int head_num, int max_thread_per_block);

template <typename T>
void ker_arrange_encdec_q_launcher(int step_token_num, int hidden_size,
                                   cudaStream_t stream, const T* ori_q,
                                   const T* q_bias, T* new_q, int beam_size,
                                   int dim_per_head, int head_num,
                                   int max_thread_per_block);

template <typename T>
void ker_correlation_softmax_encself_launcher(int batch_size, int batch_seq_len,
                                              int head_num, cudaStream_t stream,
                                              T* correlation,
                                              const int* src_padding_mask);

template <typename T>
void ker_correlation_softmax_decself_launcher(int batch_head_num, int step_num,
                                              cudaStream_t stream,
                                              T* correlation);

template <typename T>
void ker_correlation_softmax_encdec_launcher(
    int batch_size, int head_num_per_seq, int batch_seq_len,
    cudaStream_t stream, T* correlation, const int* src_padding_mask);

template <typename T>
void ker_arrange_atten_output_launcher(int batch_token_num, int hidden_size,
                                       cudaStream_t stream, const T* ori_q,
                                       T* new_q, int beam_size,
                                       int dim_per_head, int head_num,
                                       int max_thread_per_block);

__global__ void ker_refresh_result(const int* can_idx, const float* can_score,
                                   const int* num_can_per_beam,
                                   const int* old_alive_seq, int* new_alive_seq,
                                   float* seq_probs, float* seq_score,
                                   int* num_finish_beam, int vocab_size,
                                   int cur_step, float length_norm,
                                   float diverse_lambda, int end_id);

__global__ void ker_write_trg_tokenid_pos_penalty(const int* alive_seq,
                                                  float* seq_scores,
                                                  int* output, int max_step,
                                                  int beam_size);

__global__ void ker_write_trg_tokenid_neg_penalty(const int* alive_seq,
                                                  const float* seq_score,
                                                  int* output, int max_step,
                                                  int beam_size, int vocab_size,
                                                  int end_id);

__global__ void ker_write_topk_result(const int* alive_seq, float* seq_score,
                                      int* res_seq, int vocab_size,
                                      int max_step, int beam_size, int end_id);

__forceinline__ __host__ __device__ float length_norm(int length, float alpha) {
  if (alpha < 0.f) return 1.f / length;
  return pow((5.f + length) / 6.f, -alpha);
}

template <typename T>
void ker_topk_sample_launcher(int batch_size, int batch_seq_len,
                              const int max_step, int logits_seq_len,
                              int max_thread_per_block, cudaStream_t stream,
                              const T* logits, const T* logit_bias,
                              int* old_input_ids, int* new_input_ids,
                              const int vocab_size, const int k,
                              int* all_finished, curandState* curandstate,
                              int eos_id);

template <typename T>
void ker_topp_sample_launcher(int batch_size, int batch_seq_len,
                              const int max_step, int logits_seq_len,
                              int max_thread_per_block, cudaStream_t stream,
                              const T* logits, const T* logit_bias,
                              int* old_input_ids, int* new_input_ids,
                              const int vocab_size, const float p,
                              int* unfinished, curandState* curandstate,
                              int eos_id);

template <typename T>
void ker_bias_gelu_launcher(int batch_token_num, int block_dim,
                            cudaStream_t stream, T* input, const T* bias,
                            int feature_dim);

__global__ void ker_curand_setup(curandState* state);

}  // namespace cuda
}  // namespace lightseq
