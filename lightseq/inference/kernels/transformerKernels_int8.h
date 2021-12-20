#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace lightseq {
namespace cuda {

template <typename T>
void launch_quantize_tensor(const T *input, int8_t *output, int batch_tokens,
                            int hidden_size, float quant_scale,
                            cudaStream_t &stream, bool out_col32 = false);

template <typename T>
void launch_dequantize_tensor(const int32_t *input, T *output, int batch_tokens,
                              int hidden_size, float dequant_scale,
                              cudaStream_t &stream, bool in_col32 = false);

template <typename T>
void ker_norm_layer_resual_i8O_launcher(
    int token_num, int hidden_size, cudaStream_t stream, T *input,
    int8_t *output, const T *scale, const T *bias, const T *residual_bias,
    const int max_thread_per_block, float quant_scale, bool is_post_ln = false,
    bool out_col32 = false);

template <typename T>
void ker_bias_gelu_i32I_i8O_launcher(int batch_token_num, cudaStream_t stream,
                                     int32_t *input, int8_t *output,
                                     const T *bias, int feature_dim,
                                     float dequant_scale, float quant_scale,
                                     bool in_out_col32 = false);

template <typename T>
void ker_bias_gelu_i8I_i8O_launcher(int batch_token_num, cudaStream_t stream,
                                    int8_t *input, int8_t *output,
                                    const T *bias, int feature_dim,
                                    float dequant_scale, float quant_scale,
                                    bool in_out_col32 = false);

// TODO: remove clip_max
template <typename T>
void ker_bias_relu_i32I_i8O_launcher(int batch_token_num, cudaStream_t stream,
                                     int32_t *input, int8_t *output,
                                     const T *bias, int feature_dim,
                                     float dequant_scale, float quant_scale,
                                     float clip_max, bool in_out_col32 = false,
                                     bool narrow_clip = false);

// TODO: remove clip_max
template <typename T>
void ker_bias_relu_i8I_i8O_launcher(int batch_token_num, cudaStream_t stream,
                                    int8_t *input, int8_t *output,
                                    const T *bias, int feature_dim,
                                    float dequant_scale, float quant_scale,
                                    float clip_max, bool in_out_col32 = false,
                                    bool narrow_clip = false);

template <typename T>
void ker_residual_bias_ln_i32I_i8O_launcher(
    const int32_t *input, const T *scale, const T *bias, const T *residual_bias,
    int8_t *output, T *residual, int batch_tokens, int hidden_size,
    float dequant_scale, float quant_scale, int max_thread_per_block,
    cudaStream_t stream, bool is_post_ln = false, bool in_out_col32 = false,
    const T *colsum = nullptr);

template <typename T>
void ker_residual_bias_ln_i8I_i8O_launcher(
    const int8_t *input, const T *scale, const T *bias, const T *residual_bias,
    int8_t *output, T *residual, int batch_tokens, int hidden_size,
    float dequant_scale, float quant_scale, int max_thread_per_block,
    cudaStream_t stream, bool is_post_ln = false, bool in_out_col32 = false,
    const T *colsum = nullptr);

template <typename T>
void ker_residual_bias_ln_i32I_launcher(
    const int32_t *input, const T *scale, const T *bias, const T *residual,
    T *output, int batch_tokens, int hidden_size, float dequant_scale,
    int max_thread_per_block, cudaStream_t stream, bool in_col32 = false,
    const T *colsum = nullptr);

template <typename T>
void ker_residual_bias_ln_i8I_launcher(
    const int8_t *input, const T *scale, const T *bias, const T *residual,
    T *output, int batch_tokens, int hidden_size, float dequant_scale,
    int max_thread_per_block, cudaStream_t stream, bool in_col32 = false,
    const T *colsum = nullptr);

template <typename T>
void ker_arrange_encself_qkv_i32I_launcher(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const int32_t *ori_qkv, const T *qkv_bias, T *new_qkv, int max_batch_dim,
    int batch_seq_len, int dim_per_head, int head_num, int max_thread_per_block,
    float dequant_scale, bool in_col32 = false);

template <typename T>
void ker_arrange_encself_qkv_i8I_launcher(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const int8_t *ori_qkv, const T *qkv_bias, T *new_qkv, int max_batch_dim,
    int batch_seq_len, int dim_per_head, int head_num, int max_thread_per_block,
    float dequant_scale, bool in_col32 = false);

template <typename T>
void ker_arrange_atten_output_i8O_launcher(
    int batch_token_num, int hidden_size, cudaStream_t stream, const T *ori_q,
    int8_t *new_q, int beam_size, int dim_per_head, int head_num,
    int max_thread_per_block, float quant_scale, bool out_col32 = false);

template <typename T>
void ker_arrange_decself_qkv_i32I_launcher(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const int32_t *ori_qkv, const T *qkv_bias, T *new_q, T *new_k, T *new_v,
    int head_num, int dim_per_head, int max_step, int step_id,
    int max_thread_per_block, float dequant_scale, bool in_col32 = false);

template <typename T>
void ker_arrange_decself_qkv_i8I_launcher(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const int8_t *ori_qkv, const T *qkv_bias, int8_t *new_q, int8_t *new_k,
    int8_t *new_v, int head_num, int dim_per_head, int max_step, int step_id,
    int max_thread_per_block, float dequant_scale, float quant_scale,
    bool in_col32 = false);

void ker_fuse_softmax_new_value_int8_launcher(
    const int32_t *correlation, const int8_t *v, int8_t *new_v,
    int batch_head_num, int step_num, int max_step, int head_num, int head_dim,
    float attn_scale, float dequant_scale, float quant_scale, bool col32_out,
    cudaStream_t stream);

template <typename T>
void ker_arrange_encdec_q_i32I_launcher(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const int32_t *ori_q, const T *q_bias, T *new_q, int beam_size,
    int dim_per_head, int head_num, int max_thread_per_block,
    float dequant_scale, bool in_col32 = false);

template <typename T>
void ker_arrange_encdec_q_i8I_launcher(int step_token_num, int hidden_size,
                                       cudaStream_t stream, const int8_t *ori_q,
                                       const T *q_bias, T *new_q, int beam_size,
                                       int dim_per_head, int head_num,
                                       int max_thread_per_block,
                                       float dequant_scale,
                                       bool in_col32 = false);

template <typename T>
void select_beam_rough_topk_i32I_launcher(
    const int32_t *logits, const T *logit_bias, const float *seq_probs,
    const float *seq_score, const int *alive_seq, float dequant_scale,
    int *can_idx, float *can_score, int *num_beam_can, int vocab_size,
    int max_step, float length_norm, int cur_step, int step_token_num,
    int max_thread_per_block, cudaStream_t stream, int beam_size,
    float diverse_lambda, int end_id, bool in_col32 = false);

template <typename T>
void select_beam_rough_topk_i8I_launcher(
    const int8_t *logits, const T *logit_bias, const float *seq_probs,
    const float *seq_score, const int *alive_seq, float dequant_scale,
    int *can_idx, float *can_score, int *num_beam_can, int vocab_size,
    int max_step, float length_norm, int cur_step, int step_token_num,
    int max_thread_per_block, cudaStream_t stream, int beam_size,
    float diverse_lambda, int end_id, bool in_col32 = false);

}  // namespace cuda
}  // namespace lightseq
