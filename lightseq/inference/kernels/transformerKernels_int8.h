#pragma once
#include "common.h"

namespace lightseq {
namespace cuda {

template <typename T>
void launch_quantize_tensor(const T *input, int8_t *output, int total_count,
                            float scale, float clip_max, cudaStream_t &stream);

template <typename T>
void launch_dequantize_tensor(const int32_t *input, T *output, int total_count,
                              float scale, float clip_max,
                              cudaStream_t &stream);

template <typename T>
void ker_norm_layer_int8O_launcher(int token_num, int hidden_size,
                                   cudaStream_t stream, T *matrix,
                                   int8_t *output, const T *scale,
                                   const T *bias, int max_thread_per_block,
                                   float quant_scale, float clip_max);

template <typename T>
void ker_norm_layer_resual_int8O_launcher(
    int token_num, int hidden_size, cudaStream_t stream, T *input,
    int8_t *output, const T *scale, const T *bias, const T *residual_bias,
    const int max_thread_per_block, float quant_scale, float clip_max,
    bool is_post_ln = false, bool output_col32 = false);

template <typename T>
void ker_bias_gelu_int32I_int8O_launcher(int batch_token_num,
                                         cudaStream_t stream, int32_t *input,
                                         int8_t *output, const T *bias,
                                         int feature_dim, float in_scale,
                                         float in_clip_max, float out_scale,
                                         float out_clip_max);

template <typename T>
void ker_bias_relu_int32I_int8O_launcher(int batch_token_num,
                                         cudaStream_t stream, int32_t *input,
                                         int8_t *output, const T *bias,
                                         int feature_dim, float in_scale,
                                         float in_clip_max, float out_scale,
                                         float out_clip_max);

template <typename T>
void ker_residual_int32I_launcher(int32_t *input, T *output, int total_ele_num,
                                  float quant_scale, float clip_max,
                                  cudaStream_t stream);

template <typename T>
void ker_arrange_encself_qkv_int32I_launcher(
    int batch_token_num, int hidden_size, cudaStream_t stream,
    const int32_t *ori_qkv, const T *qkv_bias, T *new_qkv, int max_batch_dim,
    int batch_seq_len, int dim_per_head, int head_num, int max_thread_per_block,
    float quant_scale, float clip_max);

template <typename T>
void ker_arrange_atten_output_int8O_launcher(
    int batch_token_num, int hidden_size, cudaStream_t stream, const T *ori_q,
    int8_t *new_q, int beam_size, int dim_per_head, int head_num,
    int max_thread_per_block, float quant_scale, float clip_max);

template <typename T>
void ker_arrange_decself_qkv_int32I_launcher(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const int32_t *ori_qkv, const T *qkv_bias, T *new_q, T *new_k, T *new_v,
    int head_num, int dim_per_head, int max_step, int step_id,
    int max_thread_per_block, float quant_scale, float clip_max,
    bool input_col32);

template <typename T>
void ker_arrange_decself_qkv_int8I_launcher(
    int step_token_num, int hidden_size, cudaStream_t stream,
    const int8_t *ori_qkv, const T *qkv_bias, T *new_q, T *new_k, T *new_v,
    int head_num, int dim_per_head, int max_step, int step_id,
    int max_thread_per_block, float quant_scale, float clip_max);

template <typename T>
void ker_arrange_encdec_q_int32I_launcher(int step_token_num, int hidden_size,
                                          cudaStream_t stream,
                                          const int32_t *ori_q, const T *q_bias,
                                          T *new_q, int beam_size,
                                          int dim_per_head, int head_num,
                                          int max_thread_per_block,
                                          float quant_scale, float clip_max);

}  // namespace cuda
}  // namespace lightseq
