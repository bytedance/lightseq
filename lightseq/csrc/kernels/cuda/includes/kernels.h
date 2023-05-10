#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <vector>
#include "cmath"

#define MAX_THREADS 1024
#define WARP_SIZE 32
namespace lightseq {

enum class ActivationType { kRelu, kGelu };
enum LSLayout { kRowMajor, kColMajor, kCol32, kCOL4_4R2_8C, kCOL32_2R_4R4 };

namespace cuda {
const float kQuantRangeI8 = 127.0f;

void launch_curand_init(int total_count, int dim, cudaStream_t stream);

template <typename T>
void launch_layer_norm(T *ln_res, T *vars, T *means, const T *inp,
                       const T *scale, const T *bias, int batch_size,
                       int hidden_dim, cudaStream_t stream);

template <typename T>
void launch_layer_norm_i8(int8_t *q_out, uint8_t *clip_mask_out, T *vars,
                          T *means, const T *inp, const T *gamma,
                          const T *betta, const T *clip_max_out, int batch_size,
                          int hidden_dim, cudaStream_t stream);
template <typename T>
void launch_ln_bw(T *gamma_grad, T *betta_grad, T *inp_grad, const T *out_grad,
                  const T *residual_grad, const T *inp_or_out, const T *gamma,
                  const T *betta, const T *vars, const T *means, int batch,
                  int hidden_dim, cudaStream_t stream[2]);

template <typename T>
void launch_quant_ln_bw(T *gamma_grad, T *betta_grad, T *inp_grad, T *cmax_grad,
                        const T *out_grad, const T *residual_grad,
                        const T *inp_or_out, const T *gamma, const T *betta,
                        const T *vars, const T *means, const uint8_t *cmask,
                        int batch, int hidden_dim, cudaStream_t stream[2]);

template <typename T>
void launch_attn_softmax(T *vals, const T *attn_mask, int batch_size, int heads,
                         int from_len, int to_len, bool mask_future,
                         cudaStream_t stream);

template <typename T>
void launch_attn_softmax_bw(T *out_grad, const T *soft_inp, int rows,
                            int softmax_len, cudaStream_t stream);

template <typename T>
void launch_attn_softmax_new(T *out, T *inp, const T *attn_mask, int batch_size,
                             int heads, int from_len, int to_len, int kv_size,
                             bool mask_future, cudaStream_t stream);

template <typename T>
void launch_attn_softmax_bw_new(T *inp_grad, const T *out_grad,
                                const T *soft_inp, int rows, int softmax_len,
                                cudaStream_t stream);

//[sz0, sz1, sz2, sz3] -> [sz0, sz2, sz1, sz3]
template <typename T>
void launch_transform_0213(const T *input, T *output, int sz0, int sz1, int sz2,
                           int sz3, cudaStream_t stream);

// [b, s, 3, h] -> [3, b, nh, s, ad]
template <typename T>
void launch_bias_add_transform_20314(T *output, const T *input, const T *bias,
                                     int dim_0, int dim_1, int dim_2, int dim_3,
                                     int dim_4, cudaStream_t stream);

// [b, s, 3, h] -> 3 * [b, nh, s, ad]
template <typename T>
void launch_bias_add_transform_20314_new(T *q_out, T *k_out, T *v_out,
                                         const T *input, const T *bias,
                                         int dim_0, int dim_1, int dim_2,
                                         int dim_3, int dim_4,
                                         cudaStream_t stream);

template <typename T>
void launch_quant_bias_add_transform_20314(T *output, uint8_t *clip_mask,
                                           const int8_t *input, const T *bias,
                                           const T *clip_max, int dim_0,
                                           int dim_1, int dim_2, int dim_3,
                                           int dim_4, cudaStream_t stream,
                                           const T *out_clip_max = nullptr,
                                           bool in_col32 = false);

// [tc, b, nh, s, ad] -> [b, s, tc, nh, ad]
template <typename T>
void launch_transform4d_0213(T *output, const T *vals, int batch_size,
                             int seq_len, int hidden_dim, int nhead,
                             int trans_count, cudaStream_t stream);

template <typename T>
void launch_quant_transform4d_0213(int8_t *output, uint8_t *clip_mask,
                                   const T *vals, const T *clip_max,
                                   int batch_size, int seq_len, int hidden_dim,
                                   int nhead, int trans_count,
                                   cudaStream_t stream);
template <typename T>
void launch_transform_0213_dcmax(T *output, T *grad_cmax, const T *vals,
                                 const uint8_t *clip_mask, int batch_size,
                                 int seq_len, int hidden_dim, int nhead,
                                 cudaStream_t stream);

template <typename T>
void launch_ls_dropout(T *out, const T *vals, uint8_t *mask, int total_count,
                       float ratio, cudaStream_t stream, bool backward = false);

template <typename T>
void launch_ls_dropout_res_bias(T *out, const T *vals, uint8_t *mask,
                                const T *bias, const T *residual,
                                int total_count, int dim, float ratio,
                                cudaStream_t stream);

template <ActivationType, typename T>
void launch_ls_dropout_act_bias(T *out, const T *vals, uint8_t *mask,
                                const T *bias, int total_count, int dim,
                                float ratio, cudaStream_t stream);

template <typename T>
void launch_ls_dropout_bias_bwd(T *in_grad, T *bias_grad, const T *out_grad,
                                const uint8_t *mask, int row_size, int dim,
                                float ratio, cudaStream_t stream);

template <ActivationType act_type, typename T>
void launch_ls_dropout_act_bias_bwd(T *in_grad, T *bias_grad, const T *input,
                                    const T *bias, const T *out_grad,
                                    const uint8_t *mask, int row_size, int dim,
                                    float ratio, cudaStream_t stream);

template <ActivationType act_type, typename T>
void launch_ls_quant_dropout_act_bias(int8_t *qout, uint8_t *cmask_out,
                                      uint8_t *cmask_in, uint8_t *dropout_mask,
                                      const int8_t *qinput, const T *bias,
                                      const T *cmax_out, const T *cmax_in,
                                      int total_count, int dim, float ratio,
                                      cudaStream_t stream);

template <ActivationType act_type, typename T>
void launch_ls_fakequant_dropout_act_bias(
    T *out, uint8_t *cmask_out, uint8_t *cmask_in, uint8_t *dropout_mask,
    const int8_t *qinput, const T *bias, const T *cmax_out, const T *cmax_in,
    int total_count, int dim, float ratio, cudaStream_t stream,
    bool in_col32 = false, bool symmetry = true);

template <typename T>
void launch_ls_quant_dropout_res_bias(T *out, uint8_t *mask,
                                      const int8_t *qvals, const T *cmax,
                                      const T *bias, const T *residual,
                                      int total_count, int dim, float ratio,
                                      cudaStream_t stream,
                                      bool in_col32 = false);

template <ActivationType act_type, typename T>
void launch_ls_quant_dropout_act_bias_bwd(
    T *in_grad, T *bias_grad, T *cmax_in_grad, T *cmax_out_grad,
    const int8_t *input, const T *cmax_in, const uint8_t *cmask_in,
    const uint8_t *cmask_out, const T *bias, const T *out_grad,
    const uint8_t *dropout_mask, int row_size, int dim, float ratio,
    cudaStream_t stream, bool in_col32 = false);

template <ActivationType act_type, typename T>
void launch_ls_quant_dropout_act_bias_bwd(
    T *in_grad, T *bias_grad, T *cmax_in_grad, T *cmax_out_grad, const T *input,
    const T *cmax_in, const uint8_t *cmask_in, const uint8_t *cmask_out,
    const T *bias, const T *out_grad, const uint8_t *dropout_mask, int row_size,
    int dim, float ratio, cudaStream_t stream);

template <typename T>
void launch_fuse_transpose_bias_kernel(const T *inp, T *out, int rows, int cols,
                                       cudaStream_t stream);

void launch_param_update(const float *input, __half *output, int size,
                         cudaStream_t stream);

template <typename T>
void launch_concat3_dim1(const T *inp1, const T *inp2, T *output, int sz0,
                         int sz2, int sz1_1, int sz1_2, cudaStream_t stream);
template <typename T>
void launch_filling_concat3_dim1(T *output, const T *inp, int sz0, int mx_sz1,
                                 int sz2, int sz1_0, int sz1_1,
                                 cudaStream_t stream);

template <typename T>
void launch_fused_add2(T *out, const T *inp1, const T *inp2, int batch_size,
                       int seq_len, int hidden_size, cudaStream_t &stream);

template <typename T>
void launch_cross_entropy_fw(const T *inputs_ptr, const int *targets_ptr,
                             float *outputs_ptr, float *nll_loss_ptr,
                             float *loss_buffer, const int padding_idx,
                             const float epsilon, const int batch_size,
                             const int seq_len, const int vocab_size,
                             cudaStream_t stream);

template <typename T>
void launch_cross_entropy_bw(const float *grad_outputs_ptr, const T *inputs_ptr,
                             const int *targets_ptr, T *grad_inputs_ptr,
                             const int padding_idx, const float epsilon,
                             const int batch_size, const int seq_len,
                             const int vocab_size, cudaStream_t stream);

template <typename T>
void launch_lookup_scale_pos_dropout(
    T *output, const int *input, const T *embeddings, const T *pos_embeddings,
    const T *clip_max, uint8_t *dropout_mask, int *tokens_position,
    int batch_size, int seq_len, int embedding_dim, int padding_idx,
    float dropout_ratio, int step, cudaStream_t &stream);

template <typename T>
void launch_d_lookup_scale_pos_dropout(
    T *grad_embeddings, T *grad_clip_max, T *grad_pos_embeddings,
    const T *grad_output, const int *input, const uint8_t *dropout_mask,
    const int *tokens_position, int batch_size, int seq_len, int embedding_dim,
    int vocab_size, int max_seq_len, int padding_idx, float dropout_ratio,
    bool trainable_pos, cudaStream_t &stream);

template <typename T>
void launch_viterbi(const T *start_transition, const T *end_transition,
                    const T *transition, const T *emission, const T *mask,
                    float *best_score, int *history, int *best_tags,
                    int num_tags, int seq_len, int batch_size,
                    cudaStream_t stream, const T *bias = nullptr);

template <typename T>
void launch_quantize(int8_t *q_ptr, uint8_t *clip_mask_ptr, float *alpha_ptr,
                     const T *f_ptr, const T *clip_max_ptr, int numel,
                     int mask_start_bit, cudaStream_t stream);

template <typename T>
void launch_quantize(int8_t *q_ptr, uint8_t *clip_mask_ptr, float *alpha_ptr,
                     const T *f_ptr, const T *clip_max_ptr, int batch_tokens,
                     int hidden_size, int mask_start_bit, cudaStream_t stream,
                     LSLayout out_layout = kRowMajor);

template <typename T>
void launch_fake_quantize(uint8_t *clip_mask_ptr, float *alpha_ptr, T *output,
                          const T *input, const T *clip_max_ptr, int numel,
                          int mask_start_bit, cudaStream_t stream,
                          bool symmetry = true);

template <typename T>
void launch_dequantize(T *f_ptr, const int8_t *q_ptr, const T *clip_max_ptr,
                       int numel, int mask_start_bit, cudaStream_t stream);

template <typename T>
void launch_dequantize(T *f_ptr, const int8_t *q_ptr, const T *clip_max_ptr,
                       int batch_tokens, int hidden_size, int mask_start_bit,
                       cudaStream_t stream, bool in_col32 = false);

template <typename T>
void launch_quantize_bwd(T *grad_ptr, T *cmax_grad_ptr,
                         const uint8_t *clip_mask_ptr, int numel,
                         int mask_start_bit, cudaStream_t stream);

template <typename T>
void launch_d_cmax(T *grad_ptr, T *grad_cmax_ptr, const uint8_t *clip_mask_ptr,
                   int numel, int mask_start_bit, cudaStream_t stream);

template <typename T>
void launch_split_head(const T *inp, const T *bias, T *query, T *key, T *value,
                       int batch_size, int hidden_dim, int head_dim, int q_len,
                       int kv_len, int step, int qkv_num, cudaStream_t stream);

/* Convert 2-dim tensor index into vector index */
__forceinline__ __host__ __device__ int flat_2dim(int id1, int id2, int dim2) {
  return id1 * dim2 + id2;
}

/* Convert 3-dim tensor index into vector index */
__forceinline__ __host__ __device__ int flat_3dim(int id1, int id2, int id3,
                                                  int dim2, int dim3) {
  return id1 * dim2 * dim3 + id2 * dim3 + id3;
}

/* Convert 4-dim tensor index into vector index */
__forceinline__ __host__ __device__ int flat_4dim(int id1, int id2, int id3,
                                                  int id4, int dim2, int dim3,
                                                  int dim4) {
  // return id1*(dim2*dim3*dim4) + id2*(dim3*dim4) + id3*dim4 + id4;
  int res = id4;

  int ld = dim4;
  res += id3 * ld;

  ld *= dim3;
  res += id2 * ld;

  ld *= dim2;
  res += id1 * ld;

  return res;
}

/* Convert 5-dim tensor index into vector index */
__forceinline__ __host__ __device__ int flat_5dim(int id1, int id2, int id3,
                                                  int id4, int id5, int dim2,
                                                  int dim3, int dim4,
                                                  int dim5) {
  // return id1*(dim2*dim3*dim4*dim5) + id2*(dim3*dim4*dim5) + id3*(dim4*dim5) +
  // id4*dim5 + dim5;
  int res = id5;

  int ld = dim5;
  res += id4 * ld;

  ld *= dim4;
  res += id3 * ld;

  ld *= dim3;
  res += id2 * ld;

  ld *= dim2;
  res += id1 * ld;

  return res;
}

/* Convert 6-dim tensor index into vector index */
__forceinline__ __host__ __device__ int flat_6dim(int id1, int id2, int id3,
                                                  int id4, int id5, int id6,
                                                  int dim2, int dim3, int dim4,
                                                  int dim5, int dim6) {
  // return id1*(dim2*dim3*dim4*dim5*dim6) + id2*(dim3*dim4*dim5*dim6) +
  // id3*(dim4*dim5*dim6) + id4*(dim5*dim6) + id5*dim6 + id6;
  int res = id6;

  int ld = dim6;
  res += id5 * ld;

  ld *= dim5;
  res += id4 * ld;

  ld *= dim4;
  res += id3 * ld;

  ld *= dim3;
  res += id2 * ld;

  ld *= dim2;
  res += id1 * ld;

  return res;
}

/* Convert vector index to 6-dim tensor index */
__forceinline__ __host__ __device__ void decompose_6dim(
    int src, int dim1, int dim2, int dim3, int dim4, int dim5, int *id0,
    int *id1, int *id2, int *id3, int *id4, int *id5) {
  *id5 = src % dim5;
  src /= dim5;

  *id4 = src % dim4;
  src /= dim4;

  *id3 = src % dim3;
  src /= dim3;

  *id2 = src % dim2;
  src /= dim2;

  *id1 = src % dim1;
  *id0 = src / dim1;
}

/* Convert vector index to 5-dim tensor index */
__forceinline__ __host__ __device__ void decompose_5dim(int src, int dim1,
                                                        int dim2, int dim3,
                                                        int dim4, int *id0,
                                                        int *id1, int *id2,
                                                        int *id3, int *id4) {
  *id4 = src % dim4;
  src /= dim4;

  *id3 = src % dim3;
  src /= dim3;

  *id2 = src % dim2;
  src /= dim2;

  *id1 = src % dim1;
  *id0 = src / dim1;
}

/* Convert vector index to 4-dim tensor index */
__forceinline__ __host__ __device__ void decompose_4dim(int src, int dim1,
                                                        int dim2, int dim3,
                                                        int *id0, int *id1,
                                                        int *id2, int *id3) {
  *id3 = src % dim3;
  src /= dim3;

  *id2 = src % dim2;
  src /= dim2;

  *id1 = src % dim1;
  *id0 = src / dim1;
}

/* Convert vector index to 3-dim tensor index */
__forceinline__ __host__ __device__ void decompose_3dim(int src, int dim1,
                                                        int dim2, int *id0,
                                                        int *id1, int *id2) {
  *id2 = src % dim2;
  src /= dim2;

  *id1 = src % dim1;
  *id0 = src / dim1;
}

/* Convert vector index to 2-dim tensor index */
__forceinline__ __host__ __device__ void decompose_2dim(int src, int dim1,
                                                        int *id0, int *id1) {
  *id1 = src % dim1;
  *id0 = src / dim1;
}

__forceinline__ __device__ int get_clip_mask(float value, float clip_max,
                                             int start_bit,
                                             bool symmetry = true) {
  if (value >= clip_max) {
    return 1 << (start_bit - 1);
  } else if (symmetry && value <= -clip_max) {
    return 1 << (start_bit);
  } else {
    return 0;
  }
}

__forceinline__ __device__ int is_max_min_mask(uint8_t mask, int start_bit) {
  if (bool(mask & (1 << (start_bit - 1)))) {
    return 1;
  } else if (bool(mask & (1 << (start_bit)))) {
    return -1;
  } else {
    return 0;
  }
}

template <typename T>
__forceinline__ __device__ void clip_bwd(T &in_grad, float &cmax_grad,
                                         T out_grad, uint8_t mask,
                                         int start_bit) {
  if (bool(mask & (1 << (start_bit - 1)))) {
    in_grad = 0;
    cmax_grad = out_grad;
  } else if (bool(mask & (1 << (start_bit)))) {
    in_grad = 0;
    cmax_grad = -out_grad;
  } else {
    in_grad = out_grad;
    cmax_grad = 0;
  }
}

__forceinline__ __device__ int8_t quantize(float x, float clip_max,
                                           uint8_t &clip_mask, int start_bit) {
  clip_mask = uint8_t(get_clip_mask(x, clip_max, start_bit));
  float dequant_scale = clip_max / kQuantRangeI8;
  float i8_f = x / dequant_scale;
  float i8 = floorf(i8_f + 0.5f);
  i8 = fminf(fmaxf(i8, -kQuantRangeI8), kQuantRangeI8);
  return static_cast<int8_t>(i8);
}

__forceinline__ __device__ float fake_quantize(float x, float clip_max,
                                               uint8_t &clip_mask,
                                               int start_bit,
                                               bool symmetry = true) {
  clip_mask = uint8_t(get_clip_mask(x, clip_max, start_bit, symmetry));
  float dequant_scale = clip_max / kQuantRangeI8;
  if (!symmetry) dequant_scale /= 2;
  float i8_f = x / dequant_scale;
  if (!symmetry) i8_f -= kQuantRangeI8;
  float i8 = floorf(i8_f + 0.5f);
  i8 = fminf(fmaxf(i8, -kQuantRangeI8), kQuantRangeI8);
  if (symmetry)
    return i8 * dequant_scale;
  else
    return (i8 + kQuantRangeI8) * dequant_scale;
}

__forceinline__ __device__ float fake_quant_i8(float x, float clip_max) {
  float dequant_scale = clip_max / kQuantRangeI8;
  float i8_f = x / dequant_scale;
  float i8 = floorf(i8_f + 0.5);
  float res = fminf(fmaxf(i8, -kQuantRangeI8), kQuantRangeI8);
  return res * dequant_scale;
}

__forceinline__ __device__ float dequantize(int8_t x, float clip_max) {
  float dequant_scale = clip_max / kQuantRangeI8;
  float res = static_cast<float>(x) * dequant_scale;
  return fminf(fmaxf(res, -clip_max), clip_max);
}

/* row major index to col32 index */
__forceinline__ __host__ __device__ int row_major2flat_col32(int row_id,
                                                             int col_id,
                                                             int row_size,
                                                             int col_size) {
  return ((col_id & 0xffffe0) * row_size) + (row_id << 5) + (col_id & 31);
}

template <typename T>
__inline__ float get_float(T val);

template <>
__inline__ float get_float<float>(float val) {
  return val;
}

template <>
__inline__ float get_float<__half>(__half val) {
  return __half2float(val);
}

template <typename T>
static __global__ void zero_grad(T *grad);

template <>
__global__ void zero_grad<float>(float *grad) {
  grad[0] = 0;
}

template <>
__global__ void zero_grad<__half>(__half *grad) {
  grad[0] = __half(0.0);
}

std::string launch_gemm_test(int m, int n, int k);

__inline__ std::vector<LSLayout> getLSLayout(std::string layout) {
  if (layout == "CUBLASLT_ORDER_COL")
    return {kColMajor, kColMajor, kColMajor};
  else if (layout == "CUBLASLT_ORDER_COL4_4R2_8C")
    return {kCol32, kCOL4_4R2_8C, kCol32};
  else if (layout == "CUBLASLT_ORDER_COL32_2R_4R4")
    return {kCol32, kCOL32_2R_4R4, kCol32};
  else
    return {kRowMajor, kRowMajor, kRowMajor};
}
}  // namespace cuda
}  // namespace lightseq
