#include "encoder.h"

#include "../kernels/transformerKernels.h"
#include "../kernels/embKernels.h"
#include "../kernels/transformerKernels_int8.h"
#include "cublas_helper.h"

/**
@file
Transformer encoder, composed by gemm lib and
  custom cuda kernel function
*/

namespace lightseq {
namespace cuda {

template <OperationType OpType_>
Encoder<OpType_>::Encoder(int max_batch_size, const int *p_d_token_id,
                          int *p_d_padding_mask, _DataType *p_d_output,
                          const TransformerWeight<OpType_> &tw,
                          cudaStream_t stream, cublasHandle_t hd,
                          const int *p_d_lang_id)
    : _max_batch_size(max_batch_size),
      _p_d_token_id(p_d_token_id),
      _p_d_padding_mask(p_d_padding_mask),
      _p_d_output(p_d_output),
      _p_d_lang_id(p_d_lang_id),
      _tw(tw),
      _stream(stream),
      _hd(hd),
      _p_d_src_emb_wei(tw.get_src_emb_wei()),
      _p_d_enc_wei(tw.get_enc_wei()),
      _fone((_DataType)1.f),
      _fzero((_DataType)0.f),
#ifdef INT8_MODE
      _src_scaled_emb_clip_max(tw.get_src_scaled_emb_clip_max()),
      _enc_clip_max(tw.get_enc_clip_max()),
      _ione((int32_t)1),
      _izero((int32_t)0),
#endif
      _atten_scaler((_DataType)sqrt(1.f / tw._dim_per_head)),
      _max_batch_dim(max_batch_size * tw._max_step * tw._hidden_size),
      _max_thread_per_block(1024) {
  CHECK_GPU_ERROR(cublasLtCreate(&_cublas_lt_handle));
}

/**
Compute GPU memory size needed by transformer encoder,
  to see how these memory is used, checkout init_buffer() for detail
*/
template <OperationType OpType_>
long Encoder<OpType_>::compute_buffer_bytesize() {
  long sz1 = _max_batch_dim * 6 +
             _max_batch_size * _tw._head_num * _tw._max_step * _tw._max_step;
  long sz2 = _max_batch_dim + _max_batch_size * _tw._max_step * _tw._inner_size;
  return max(sz1, sz2) * sizeof(_DataType);
}

/**
Init the GPU memory pointer which point to
  the memory buffer needed by encoder.
These buffer are used during custom cuda kernel function,
  find the corresponding function to see how these buffer are used
*/
template <OperationType OpType_>
void Encoder<OpType_>::init_buffer(void *pbuf) {
  _DataType *p_d_buf = reinterpret_cast<_DataType *>(pbuf);
  _p_d_qkv_projected = p_d_buf;
  _p_d_q = _p_d_qkv_projected + _max_batch_dim * 3;
  _p_d_k = _p_d_q + _max_batch_dim;
  _p_d_v = _p_d_k + _max_batch_dim;
  _p_d_c = _p_d_v + _max_batch_dim;
  _p_d_ffn_buf1 = p_d_buf;
  _p_d_ffn_buf2 = _p_d_ffn_buf1 + _max_batch_dim;
  // encoder and decoder use the same buffer to save gpu memory useage
#ifdef INT8_MODE
  int max_batch_dim = _max_batch_size * _tw._max_step *
                      std::max(_tw._inner_size, _tw._hidden_size * 3);
  CHECK_GPU_ERROR(cudaMalloc(&_int8_ffn_in_buf, max_batch_dim));
  CHECK_GPU_ERROR(
      cudaMalloc(&_int32_ffn_out_buf, max_batch_dim * sizeof(int32_t)));
  _int8_p_d_enc_wei = std::vector<int8_t *>(_tw._n_enc_layer * 4);
  _scaled_ffn2_colsum = std::vector<_DataType *>(_tw._n_enc_layer);
  for (_layer_id = 0; _layer_id < _tw._n_enc_layer; _layer_id++) {
    _weight_offset = _layer_id * _tw._weight_per_enc_layer;
    CHECK_GPU_ERROR(cudaMalloc(&_int8_p_d_enc_wei[_layer_id * 4],
                               _tw._hidden_size * 3 * _tw._hidden_size));
    CHECK_GPU_ERROR(cudaMalloc(&_int8_p_d_enc_wei[_layer_id * 4 + 1],
                               _tw._hidden_size * _tw._hidden_size));
    CHECK_GPU_ERROR(cudaMalloc(&_int8_p_d_enc_wei[_layer_id * 4 + 2],
                               _tw._hidden_size * _tw._inner_size));
    CHECK_GPU_ERROR(cudaMalloc(&_int8_p_d_enc_wei[_layer_id * 4 + 3],
                               _tw._inner_size * _tw._hidden_size));

    quantize_weight_col32t(
        _p_d_enc_wei[_weight_offset + 2], _int8_p_d_enc_wei[_layer_id * 4],
        _tw._hidden_size, _tw._hidden_size * 3, _quant_scale,
        _enc_clip_max[_layer_id * 12], _stream, _cublas_lt_handle);

    quantize_weight_col32t(
        _p_d_enc_wei[_weight_offset + 4], _int8_p_d_enc_wei[_layer_id * 4 + 1],
        _tw._hidden_size, _tw._hidden_size, _quant_scale,
        _enc_clip_max[_layer_id * 12 + 1], _stream, _cublas_lt_handle);

    quantize_weight_col32t(
        _p_d_enc_wei[_weight_offset + 8], _int8_p_d_enc_wei[_layer_id * 4 + 2],
        _tw._hidden_size, _tw._inner_size, _quant_scale,
        _enc_clip_max[_layer_id * 12 + 2], _stream, _cublas_lt_handle);

    quantize_weight_col32t(
        _p_d_enc_wei[_weight_offset + 10], _int8_p_d_enc_wei[_layer_id * 4 + 3],
        _tw._inner_size, _tw._hidden_size, _quant_scale,
        _enc_clip_max[_layer_id * 12 + 3], _stream, _cublas_lt_handle);

    if (_tw._use_gelu) {
      _scaled_ffn2_colsum[_layer_id] = nullptr;
    } else {
      CHECK_GPU_ERROR(cudaMalloc(&_scaled_ffn2_colsum[_layer_id],
                                 _tw._hidden_size * sizeof(_DataType)));
      launch_scaled_colsum(_p_d_enc_wei[_weight_offset + 10],
                           _scaled_ffn2_colsum[_layer_id], _tw._inner_size,
                           _tw._hidden_size,
                           _enc_clip_max[_layer_id * 12 + 7] / 2, _stream);
    }
  }
#endif
  return;
}

/**
Some requirements needed by custom cuda kernel function
*/
template <OperationType OpType_>
std::string Encoder<OpType_>::check() {
  // if (_max_thread_per_block < _tw._hidden_size) {
  //   return "violate hidden_size <= max_thread_per_block";
  // }
  if (_tw._inner_size & 1) {
    return "violate inner_size % 2 = 0";
  }
  if (_tw._dim_per_head & 1) {
    return "violate dim_per_head % 2 = 0";
  }
  if (_tw._multilg_type == 0 && _p_d_src_emb_wei.size() != 4) {
    return "violate p_d_src_emb_wei.size() = 4";
  }
  if (_tw._multilg_type != 0 && _p_d_src_emb_wei.size() != 5) {
    return "violate p_d_src_emb_wei.size() = 5";
  }
  if (_p_d_enc_wei.size() != _tw._weight_per_enc_layer * _tw._n_enc_layer) {
    return "violate p_d_enc_wei.size() = weight_per_enc_layer * n_enc_layer";
  }
  if (_tw._multilg_type != 0 && _p_d_lang_id == nullptr) {
    return "lang id should not be null when multilg";
  }
  return "";
}

/**
Encoder inference
*/
template <OperationType OpType_>
void Encoder<OpType_>::run_one_infer(int batch_size, int batch_seq_len) {
  /* ---step1. init--- */
  _batch_size = batch_size;
  _batch_seq_len = batch_seq_len;
  _batch_token_num = batch_size * batch_seq_len;
#ifdef DEBUG_RESULT
  std::cout << "batch_size-" << batch_size << " batch_seq_len-" << batch_seq_len
            << std::endl;
  print_vec(_p_d_token_id, "batch_token_ids", batch_size * batch_seq_len);
#endif

  /* ---step2. encoder feedforward--- */
  launch_enc_emb<_DataType>(_p_d_src_emb_wei[0], _p_d_src_emb_wei[1],
                            _p_d_token_id, _p_d_output, _p_d_padding_mask,
                            _tw._padding_id, batch_size, batch_seq_len,
                            _tw._hidden_size, _stream, _p_d_src_emb_wei[4],
                            _p_d_lang_id, _tw._multilg_type);
#ifdef DEBUG_RESULT
  for (int i = 0; i < _batch_size; i++) {       // batch_id
    for (int j = 0; j < _batch_seq_len; j++) {  // token_id
      std::cout << "emb out: token-" << j << std::endl;
      print_vec(_p_d_output + i * _batch_seq_len * _tw._hidden_size +
                    j * _tw._hidden_size,
                "emb out", 10);
    }
  }  // not normal
#endif
  for (_layer_id = 0; _layer_id < _tw._n_enc_layer; _layer_id++) {
    _weight_offset = _layer_id * _tw._weight_per_enc_layer;
    self_attention();
    ffn_add_norm();
  }

#ifndef INT8_MODE
  // last layer norm
  ker_norm_layer_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_output,
      _p_d_src_emb_wei[2], _p_d_src_emb_wei[3], _max_thread_per_block);
#endif

#ifdef DEBUG_RESULT
  for (int i = 0; i < _batch_size; i++) {       // batch_id
    for (int j = 0; j < _batch_seq_len; j++) {  // token_id
      std::cout << "encoder output: token-" << j << std::endl;
      print_vec(_p_d_output + i * _batch_seq_len * _tw._hidden_size +
                    j * _tw._hidden_size,
                "encoder_output", _tw._dim_per_head);
    }
  }  // not normal
#endif
  return;
}

/**
Encoder self attention
*/
template <OperationType OpType_>
void Encoder<OpType_>::self_attention() {
#ifdef INT8_MODE
  if (_layer_id == 0) {
    ker_norm_layer_resual_int8O_launcher<_DataType>(
        _batch_token_num, _tw._hidden_size, _stream, _p_d_output,
        _int8_ffn_in_buf, _p_d_enc_wei[_weight_offset],
        _p_d_enc_wei[_weight_offset + 1], _p_d_enc_wei[_weight_offset + 5],
        _max_thread_per_block, _quant_scale, _enc_clip_max[_layer_id * 12 + 4],
        _tw._is_post_ln, true);
  }

  cublasLtMM_withAlgo(_int32_ffn_out_buf, 1, _batch_token_num,
                      _tw._hidden_size * 3, _tw._hidden_size, 0, 0, 0,
                      _int8_ffn_in_buf, _int8_p_d_enc_wei[_layer_id * 4],
                      _cublas_lt_handle, _stream, false);

  // get q, k, v by split and reshape qkv
  ker_arrange_encself_qkv_int32I_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _int32_ffn_out_buf,
      _p_d_enc_wei[_weight_offset + 3], _p_d_q, _max_batch_dim, _batch_seq_len,
      _tw._dim_per_head, _tw._head_num, _max_thread_per_block,
      _quant_scale * _quant_scale,
      _enc_clip_max[_layer_id * 12] * _enc_clip_max[_layer_id * 12 + 4], true);
#else
  /* ---step 0. layer_norm, add output_bias to "query"--- */
  ker_norm_layer_resual_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_output, _p_d_q,
      _p_d_enc_wei[_weight_offset], _p_d_enc_wei[_weight_offset + 1],
      _p_d_enc_wei[_weight_offset + 5], _max_thread_per_block, _tw._is_post_ln);

  /* ---step 1. qkv = ori_q * qkv_wei + bias, and reshape qkv for multi-head
   * gemm--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size * 3, _batch_token_num,
      _tw._hidden_size, &_fone, _p_d_enc_wei[_weight_offset + 2], _AType,
      _tw._hidden_size * 3, _p_d_q, _BType, _tw._hidden_size, &_fzero,
      _p_d_qkv_projected, _CType, _tw._hidden_size * 3, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  // get q, k, v by split and reshape qkv
  ker_arrange_encself_qkv_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_qkv_projected,
      _p_d_enc_wei[_weight_offset + 3], _p_d_q, _max_batch_dim, _batch_seq_len,
      _tw._dim_per_head, _tw._head_num, _max_thread_per_block);
#endif

  /* ---step 2. correlation = q * k, perform softmax on correlation--- */
  CHECK_GPU_ERROR(cublasGemmStridedBatchedEx(
      _hd, CUBLAS_OP_T, CUBLAS_OP_N, _batch_seq_len, _batch_seq_len,
      _tw._dim_per_head, &_atten_scaler, _p_d_k, _AType, _tw._dim_per_head,
      _batch_seq_len * _tw._dim_per_head, _p_d_q, _BType, _tw._dim_per_head,
      _batch_seq_len * _tw._dim_per_head, &_fzero, _p_d_c, _CType,
      _batch_seq_len, _batch_seq_len * _batch_seq_len,
      _batch_size * _tw._head_num, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  ker_correlation_softmax_encself_launcher<_DataType>(
      _batch_size, _batch_seq_len, _tw._head_num, _stream, _p_d_c,
      _p_d_padding_mask);

  /* ---step 3. new_q = correlation * v--- */
  CHECK_GPU_ERROR(cublasGemmStridedBatchedEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._dim_per_head, _batch_seq_len,
      _batch_seq_len, &_fone, _p_d_v, _AType, _tw._dim_per_head,
      _batch_seq_len * _tw._dim_per_head, _p_d_c, _BType, _batch_seq_len,
      _batch_seq_len * _batch_seq_len, &_fzero, _p_d_q, _CType,
      _tw._dim_per_head, _batch_seq_len * _tw._dim_per_head,
      _batch_size * _tw._head_num, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#ifdef INT8_MODE
  // use v to save reshaped q, since they are in same size and v
  // will not be use again before the next multi-head-attention
  ker_arrange_atten_output_int8O_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_q, _int8_ffn_in_buf,
      _batch_seq_len, _tw._dim_per_head, _tw._head_num, _max_thread_per_block,
      _quant_scale, _enc_clip_max[_layer_id * 12 + 5], true);

  /* ---step 4. new_q = ori_q + new_q * output_wei--- */
  cublasLtMM_withAlgo(_int32_ffn_out_buf, 1, _batch_token_num, _tw._hidden_size,
                      _tw._hidden_size, 0, 0, 0, _int8_ffn_in_buf,
                      _int8_p_d_enc_wei[_layer_id * 4 + 1], _cublas_lt_handle,
                      _stream, false);

  ker_residual_bias_ln_i32I_i8O_launcher<_DataType>(
      _int32_ffn_out_buf, _p_d_enc_wei[_weight_offset + 6],
      _p_d_enc_wei[_weight_offset + 7], _p_d_enc_wei[_weight_offset + 11],
      _int8_ffn_in_buf, _p_d_output, _batch_token_num, _tw._hidden_size,
      _enc_clip_max[_layer_id * 12 + 1] * _enc_clip_max[_layer_id * 12 + 5] /
          (_quant_scale * _quant_scale),
      _quant_scale, _enc_clip_max[_layer_id * 12 + 6], _max_thread_per_block,
      _stream, _tw._is_post_ln, true);

#else
  // use v to save reshaped q, since they are in same size and v
  // will not be use again before the next multi-head-attention
  ker_arrange_atten_output_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_q, _p_d_v,
      _batch_seq_len, _tw._dim_per_head, _tw._head_num, _max_thread_per_block);

  /* ---step 4. new_q = ori_q + new_q * output_wei--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size, _batch_token_num,
      _tw._hidden_size, &_fone, _p_d_enc_wei[_weight_offset + 4], _AType,
      _tw._hidden_size, _p_d_v, _BType, _tw._hidden_size, &_fone, _p_d_output,
      _CType, _tw._hidden_size, _computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#endif
  return;
}

template <OperationType OpType_>
void Encoder<OpType_>::ffn_add_norm() {
#ifdef INT8_MODE

  cublasLtMM_withAlgo(_int32_ffn_out_buf, 1, _batch_token_num, _tw._inner_size,
                      _tw._hidden_size, 0, 0, 0, _int8_ffn_in_buf,
                      _int8_p_d_enc_wei[_layer_id * 4 + 2], _cublas_lt_handle,
                      _stream, false);

  if (_tw._use_gelu) {
    ker_bias_gelu_int32I_int8O_launcher<_DataType>(
        _batch_token_num, _stream, _int32_ffn_out_buf, _int8_ffn_in_buf,
        _p_d_enc_wei[_weight_offset + 9], _tw._inner_size,
        _quant_scale * _quant_scale,
        _enc_clip_max[_layer_id * 12 + 2] * _enc_clip_max[_layer_id * 12 + 6],
        _quant_scale, _enc_clip_max[_layer_id * 12 + 7], true);
  } else {
    ker_bias_relu_int32I_int8O_launcher<_DataType>(
        _batch_token_num, _stream, _int32_ffn_out_buf, _int8_ffn_in_buf,
        _p_d_enc_wei[_weight_offset + 9], _tw._inner_size,
        _quant_scale * _quant_scale,
        _enc_clip_max[_layer_id * 12 + 2] * _enc_clip_max[_layer_id * 12 + 6],
        _quant_scale, _enc_clip_max[_layer_id * 12 + 7], true, true);
  }
  /* ---step 2. second ffn layer--- */
  cublasLtMM_withAlgo(_int32_ffn_out_buf, 1, _batch_token_num, _tw._hidden_size,
                      _tw._inner_size, 0, 0, 0, _int8_ffn_in_buf,
                      _int8_p_d_enc_wei[_layer_id * 4 + 3], _cublas_lt_handle,
                      _stream, false);

  const _DataType *scale_ptr, *bias_ptr, *res_bias_ptr;
  float clip_max;
  if (_layer_id == _tw._n_enc_layer - 1) {
    scale_ptr = _p_d_src_emb_wei[2];
    bias_ptr = _p_d_src_emb_wei[3];

    ker_residual_bias_ln_i32I_launcher<_DataType>(
        _int32_ffn_out_buf, scale_ptr, bias_ptr, _p_d_output, _p_d_output,
        _batch_token_num, _tw._hidden_size,
        _enc_clip_max[_layer_id * 12 + 3] * _enc_clip_max[_layer_id * 12 + 7] /
            (2 * _quant_scale * _quant_scale),
        _max_thread_per_block, _stream, true, _scaled_ffn2_colsum[_layer_id]);
  } else {
    scale_ptr = _p_d_enc_wei[(_layer_id + 1) * _tw._weight_per_enc_layer];
    bias_ptr = _p_d_enc_wei[(_layer_id + 1) * _tw._weight_per_enc_layer + 1];
    res_bias_ptr =
        _p_d_enc_wei[(_layer_id + 1) * _tw._weight_per_enc_layer + 5];
    clip_max = _enc_clip_max[(_layer_id + 1) * 12 + 4];
    ker_residual_bias_ln_i32I_i8O_launcher<_DataType>(
        _int32_ffn_out_buf, scale_ptr, bias_ptr, res_bias_ptr, _int8_ffn_in_buf,
        _p_d_output, _batch_token_num, _tw._hidden_size,
        _enc_clip_max[_layer_id * 12 + 3] * _enc_clip_max[_layer_id * 12 + 7] /
            (2 * _quant_scale * _quant_scale),
        _quant_scale, clip_max, _max_thread_per_block, _stream, _tw._is_post_ln,
        true, _scaled_ffn2_colsum[_layer_id]);
  }
#else
  /* ---step 0. layer_norm, add output_bias to "query"--- */
  ker_norm_layer_resual_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_output, _p_d_ffn_buf1,
      _p_d_enc_wei[_weight_offset + 6], _p_d_enc_wei[_weight_offset + 7],
      _p_d_enc_wei[_weight_offset + 11], _max_thread_per_block,
      _tw._is_post_ln);
  /* ---step 1. first ffn layer--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._inner_size, _batch_token_num,
      _tw._hidden_size, &_fone, _p_d_enc_wei[_weight_offset + 8], _AType,
      _tw._inner_size, _p_d_ffn_buf1, _BType, _tw._hidden_size, &_fzero,
      _p_d_ffn_buf2, _CType, _tw._inner_size, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  if (_tw._use_gelu) {
    ker_bias_gelu_launcher<_DataType>(
        _batch_token_num, _max_thread_per_block, _stream, _p_d_ffn_buf2,
        _p_d_enc_wei[_weight_offset + 9], _tw._inner_size);
  } else {
    ker_bias_relu_launcher<_DataType>(
        _batch_token_num, _max_thread_per_block, _stream, _p_d_ffn_buf2,
        _p_d_enc_wei[_weight_offset + 9], _tw._inner_size);
  }
  /* ---step 2. second ffn layer--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size, _batch_token_num,
      _tw._inner_size, &_fone, _p_d_enc_wei[_weight_offset + 10], _AType,
      _tw._hidden_size, _p_d_ffn_buf2, _BType, _tw._inner_size, &_fone,
      _p_d_output, _CType, _tw._hidden_size, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#endif
  return;
}

template class Encoder<OperationType::FP16>;
template class Encoder<OperationType::FP32>;

}  // namespace cuda
}  // namespace lightseq
