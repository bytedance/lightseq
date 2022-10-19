#include "quant_vit_encoder.h"
#include "../kernels/embKernels.h"
#include "../kernels/transformerKernels.h"
#include "../kernels/transformerKernels_int8.h"
#include "cublas_helper.h"

/**
@file
QuantViT encoder, composed by gemm lib and
  custom cuda kernel function
*/

namespace lightseq {
namespace cuda {

template <OperationType OpType_>
QuantVitEncoder<OpType_>::QuantVitEncoder(
    int max_batch_size, const float *p_d_pixel_input, int *p_d_padding_mask,
    _DataType *p_d_output, const QuantVitWeight<OpType_> &tw,
    cudaStream_t stream, cublasHandle_t hd)
    : _max_batch_size(max_batch_size),
      _p_d_pixel_input(p_d_pixel_input),
      _p_d_padding_mask(p_d_padding_mask),
      _p_d_output(p_d_output),
      _tw(tw),
      _stream(stream),
      _hd(hd),
      _p_d_src_emb_wei(tw.get_src_emb_wei()),
      _p_d_enc_wei(tw.get_enc_wei()),
      _fone((_DataType)1.f),
      _fzero((_DataType)0.f),
      _enc_clip_max(tw.get_enc_clip_max()),
      _ione((int32_t)1),
      _izero((int32_t)0),
      _atten_scaler((_DataType)sqrt(1.f / tw._dim_per_head)),
      _max_batch_dim(max_batch_size * tw._max_step * tw._hidden_size),
      _max_thread_per_block(1024),
      _sm_gt_eq_80(getSMVersion() >= 80 ? true : false) {
  CHECK_GPU_ERROR(cublasLtCreate(&_cublas_lt_handle));
}

/**
Init the GPU memory pointer which point to
  the memory buffer needed by encoder.
These buffer are used during custom cuda kernel function,
  find the corresponding function to see how these buffer are used
*/
template <OperationType OpType_>
void QuantVitEncoder<OpType_>::init_buffer() {
  std::cout << "encoder buffer init start" << std::endl;

  _DataType *qkv_buf;
  CHECK_GPU_ERROR(cudaMalloc(&qkv_buf, 3 * _max_batch_dim * sizeof(_DataType)));
  _p_d_q = qkv_buf;
  _p_d_k = qkv_buf + _max_batch_dim;
  _p_d_v = qkv_buf + 2 * _max_batch_dim;

  CHECK_GPU_ERROR(cudaMalloc(&_p_d_c, _max_batch_size * _tw._head_num *
                                          _tw._max_step * _tw._max_step *
                                          sizeof(_DataType)));

  int max_batch_dim = _max_batch_size * _tw._max_step *
                      std::max(_tw._inner_size, _tw._hidden_size * 3);
  CHECK_GPU_ERROR(cudaMalloc(&_int8_ffn_in_buf, max_batch_dim));
  CHECK_GPU_ERROR(
      cudaMalloc(&_int32_ffn_out_buf, max_batch_dim * sizeof(int32_t)));
  CHECK_GPU_ERROR(
      cudaMalloc(&_int8_ffn_out_buf, max_batch_dim * sizeof(int8_t)));

  // prepare gpu memory for weight
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

    _p_device_wei.push_back(
        to_gpu(_p_d_enc_wei[_weight_offset], _tw._hidden_size, _stream));
    _p_device_wei.push_back(
        to_gpu(_p_d_enc_wei[_weight_offset + 1], _tw._hidden_size, _stream));
    _p_device_wei.push_back(nullptr);
    _p_device_wei.push_back(to_gpu(_p_d_enc_wei[_weight_offset + 3],
                                   _tw._hidden_size * 3, _stream));
    _p_device_wei.push_back(nullptr);
    _p_device_wei.push_back(
        to_gpu(_p_d_enc_wei[_weight_offset + 5], _tw._hidden_size, _stream));
    _p_device_wei.push_back(
        to_gpu(_p_d_enc_wei[_weight_offset + 6], _tw._hidden_size, _stream));
    _p_device_wei.push_back(
        to_gpu(_p_d_enc_wei[_weight_offset + 7], _tw._hidden_size, _stream));
    _p_device_wei.push_back(nullptr);
    _p_device_wei.push_back(
        to_gpu(_p_d_enc_wei[_weight_offset + 9], _tw._inner_size, _stream));
    _p_device_wei.push_back(nullptr);
    _p_device_wei.push_back(
        to_gpu(_p_d_enc_wei[_weight_offset + 11], _tw._hidden_size, _stream));

    quantize_weight(_p_d_enc_wei[_weight_offset + 2],
                    _int8_p_d_enc_wei[_layer_id * 4], _tw._hidden_size,
                    _tw._hidden_size * 3,
                    _quant_range / _enc_clip_max[_layer_id * 11], _stream,
                    _cublas_lt_handle);

    quantize_weight(_p_d_enc_wei[_weight_offset + 4],
                    _int8_p_d_enc_wei[_layer_id * 4 + 1], _tw._hidden_size,
                    _tw._hidden_size,
                    _quant_range / _enc_clip_max[_layer_id * 11 + 1], _stream,
                    _cublas_lt_handle);

    quantize_weight(_p_d_enc_wei[_weight_offset + 8],
                    _int8_p_d_enc_wei[_layer_id * 4 + 2], _tw._hidden_size,
                    _tw._inner_size,
                    _quant_range / _enc_clip_max[_layer_id * 11 + 2], _stream,
                    _cublas_lt_handle);

    quantize_weight(_p_d_enc_wei[_weight_offset + 10],
                    _int8_p_d_enc_wei[_layer_id * 4 + 3], _tw._inner_size,
                    _tw._hidden_size,
                    _quant_range / _enc_clip_max[_layer_id * 11 + 3], _stream,
                    _cublas_lt_handle);

    if (_tw._use_gelu) {
      _scaled_ffn2_colsum[_layer_id] = nullptr;
    } else {
      CHECK_GPU_ERROR(cudaMalloc(&_scaled_ffn2_colsum[_layer_id],
                                 _tw._hidden_size * sizeof(_DataType)));
      float relu_scale = _enc_clip_max[_layer_id * 11 + 7] / 2;
      _DataType *temp;
      int weight_size = _tw._inner_size * _tw._hidden_size;

      CHECK_GPU_ERROR(cudaMalloc(&temp, weight_size * sizeof(_DataType)));
      CHECK_GPU_ERROR(cudaMemcpyAsync(temp, _p_d_enc_wei[_weight_offset + 10],
                                      weight_size * sizeof(_DataType),
                                      cudaMemcpyHostToDevice, _stream));
      launch_scaled_colsum(temp, _scaled_ffn2_colsum[_layer_id],
                           _tw._inner_size, _tw._hidden_size, relu_scale,
                           _stream);

      CHECK_GPU_ERROR(cudaGetLastError());
      CHECK_GPU_ERROR(cudaFree(temp));
    }
  }
  std::cout << "encoder buffer init succeed" << std::endl;
  return;
}

/**
Some requirements needed by custom cuda kernel function
*/
template <OperationType OpType_>
std::string QuantVitEncoder<OpType_>::check() {
  // if (_max_thread_per_block < _tw._hidden_size) {
  //   return "violate hidden_size <= max_thread_per_block";
  // }
  if (_tw._inner_size & 1) {
    return "violate inner_size % 2 = 0";
  }
  if (_tw._dim_per_head & 1) {
    return "violate dim_per_head % 2 = 0";
  }
  if (_p_d_src_emb_wei.size() != 6) {
    return "violate p_d_src_emb_wei.size() = 6";
  }
  if (_p_d_enc_wei.size() != _tw._weight_per_enc_layer * _tw._n_enc_layer) {
    return "violate p_d_enc_wei.size() = weight_per_enc_layer * n_enc_layer";
  }
  if (_tw._max_step != (_tw._image_size / _tw._patch_size) *
                               (_tw._image_size / _tw._patch_size) +
                           1) {
    return "violate max_step = (image_size / patch_size) ** 2 + 1";
  }
  return "";
}

/**
Encoder inference
*/
template <OperationType OpType_>
void QuantVitEncoder<OpType_>::run_one_infer(int batch_size) {
  if (batch_size > _max_batch_size) {
    throw std::runtime_error("batch size of input greater than max_batch_size");
  }
  /* ---step1. init--- */
  _batch_size = batch_size;
  _batch_seq_len = _tw._max_step;
  _batch_token_num = batch_size * _batch_seq_len;

  /* ---step2. encoder feedforward--- */
  launch_patch_emb<_DataType>(_p_d_src_emb_wei[0], _p_d_src_emb_wei[1],
                              _p_d_src_emb_wei[2], _p_d_src_emb_wei[3],
                              _p_d_pixel_input, _p_d_output, _tw._patch_size,
                              _tw._image_size, _batch_size, _tw._max_step,
                              _tw._hidden_size, _tw._channel_input, _stream);
#ifdef DEBUG_RESULT
  for (int i = 0; i < _batch_size; i++) {  // batch_id
    for (int j = 0; j < 10; j++) {         // patch_id
      std::cout << "emb out: patch-" << j << std::endl;
      print_vec(_p_d_output + i * _batch_seq_len * _tw._hidden_size +
                    j * _tw._hidden_size,
                "emb out", 20);
    }
  }
#endif
  for (_layer_id = 0; _layer_id < _tw._n_enc_layer; _layer_id++) {
    _weight_offset = _layer_id * _tw._weight_per_enc_layer;
    self_attention();
    ffn_add_norm();
  }

#ifdef DEBUG_RESULT
  for (int i = 0; i < _batch_size; i++) {       // batch_id
    for (int j = 0; j < _batch_seq_len; j++) {  // patch_id
      std::cout << "encoder output: token-" << j << std::endl;
      print_vec(_p_d_output + i * _batch_seq_len * _tw._hidden_size +
                    j * _tw._hidden_size,
                "encoder_output", _tw._dim_per_head);
    }
  }
#endif
  return;
}

/**
Encoder self attention
*/
template <OperationType OpType_>
void QuantVitEncoder<OpType_>::self_attention() {
  /* ---step 0. layer_norm, add output_bias to "query"--- */
  if (_layer_id == 0) {
    ker_norm_layer_resual_i8O_launcher<_DataType>(
        _batch_token_num, _tw._hidden_size, _stream, _p_d_output,
        _int8_ffn_in_buf, _p_device_wei[_weight_offset],
        _p_device_wei[_weight_offset + 1], _p_device_wei[_weight_offset + 5],
        _max_thread_per_block, _quant_range / _enc_clip_max[_layer_id * 11 + 4],
        _tw._is_post_ln, true);
  }
  CHECK_GPU_ERROR(cudaGetLastError());

#ifdef DEBUG_RESULT
  for (int i = 0; i < _batch_size; i++) {       // batch_id
    for (int j = 0; j < _batch_seq_len; j++) {  // token_id
      std::cout << "qkv_attn input: token-" << j << std::endl;
      print_vec(_int8_ffn_in_buf + i * _batch_seq_len * _tw._hidden_size +
                    j * _tw._hidden_size,
                "qkv_attn input", 10);
    }
  }
#endif

  /* ---step 1. qkv = ori_q * qkv_wei + bias, and reshape qkv for multi-head
   * gemm--- */
  cublasLtMM_withAlgo_i8IO(
      _int8_ffn_out_buf, 1, _batch_token_num, _tw._hidden_size * 3,
      _tw._hidden_size, 0, 0, 0,
      _enc_clip_max[_layer_id * 11] * _enc_clip_max[_layer_id * 11 + 4] /
          (_enc_clip_max[_layer_id * 11 + 8] * _quant_range),
      _int8_ffn_in_buf, _int8_p_d_enc_wei[_layer_id * 4], _cublas_lt_handle,
      _stream, _sm_gt_eq_80);

  // get q, k, v by split and reshape qkv
  ker_arrange_encself_qkv_i8I_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _int8_ffn_out_buf,
      _p_device_wei[_weight_offset + 3], _p_d_q, _max_batch_dim, _batch_seq_len,
      _tw._dim_per_head, _tw._head_num, _max_thread_per_block,
      _enc_clip_max[_layer_id * 11 + 8] / _quant_range, true);

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
  // use v to save reshaped q, since they are in same size and v
  // will not be use again before the next multi-head-attention
  ker_arrange_atten_output_i8O_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_q, _int8_ffn_in_buf,
      _batch_seq_len, _tw._dim_per_head, _tw._head_num, _max_thread_per_block,
      _quant_range / _enc_clip_max[_layer_id * 11 + 5], true);

#ifdef DEBUG_RESULT
  for (int i = 0; i < _batch_size; i++) {       // batch_id
    for (int j = 0; j < _batch_seq_len; j++) {  // token_id
      std::cout << "out_attn input: token-" << j << std::endl;
      print_vec(_int8_ffn_in_buf + i * _batch_seq_len * _tw._hidden_size +
                    j * _tw._hidden_size,
                "out_attn input", 10);
    }
  }
#endif

  /* ---step 4. new_q = ori_q + new_q * output_wei--- */
  cublasLtMM_withAlgo_i8IO(
      _int8_ffn_out_buf, 1, _batch_token_num, _tw._hidden_size,
      _tw._hidden_size, 0, 0, 0,
      _enc_clip_max[_layer_id * 11 + 1] * _enc_clip_max[_layer_id * 11 + 5] /
          (_enc_clip_max[_layer_id * 11 + 9] * _quant_range),
      _int8_ffn_in_buf, _int8_p_d_enc_wei[_layer_id * 4 + 1], _cublas_lt_handle,
      _stream, _sm_gt_eq_80);

#ifdef DEBUG_RESULT
  for (int i = 0; i < _batch_size; i++) {       // batch_id
    for (int j = 0; j < _batch_seq_len; j++) {  // token_id
      std::cout << "attn_ln input: token-" << j << std::endl;
      print_vec(_int8_ffn_out_buf + i * _batch_seq_len * _tw._hidden_size +
                    j * _tw._hidden_size,
                "attn_ln input", 10);
    }
  }
#endif

  ker_residual_bias_ln_i8I_i8O_launcher<_DataType>(
      _int8_ffn_out_buf, _p_device_wei[_weight_offset + 6],
      _p_device_wei[_weight_offset + 7], _p_device_wei[_weight_offset + 11],
      _int8_ffn_in_buf, _p_d_output, _batch_token_num, _tw._hidden_size,
      _enc_clip_max[_layer_id * 11 + 9] / _quant_range,
      _quant_range / _enc_clip_max[_layer_id * 11 + 6], _max_thread_per_block,
      _stream, _tw._is_post_ln, true, true);

  return;
}

template <OperationType OpType_>
void QuantVitEncoder<OpType_>::ffn_add_norm() {
#ifdef DEBUG_RESULT
  for (int i = 0; i < _batch_size; i++) {       // batch_id
    for (int j = 0; j < _batch_seq_len; j++) {  // token_id
      std::cout << "ffn1 input: token-" << j << std::endl;
      print_vec(_int8_ffn_in_buf + i * _batch_seq_len * _tw._hidden_size +
                    j * _tw._hidden_size,
                "ffn1 input", 10);
    }
  }
#endif

  /* ---step 1. first ffn layer--- */
  cublasLtMM_withAlgo_i8IO(
      _int8_ffn_out_buf, 1, _batch_token_num, _tw._inner_size, _tw._hidden_size,
      0, 0, 0,
      _enc_clip_max[_layer_id * 11 + 2] * _enc_clip_max[_layer_id * 11 + 6] /
          (_enc_clip_max[_layer_id * 11 + 10] * _quant_range),
      _int8_ffn_in_buf, _int8_p_d_enc_wei[_layer_id * 4 + 2], _cublas_lt_handle,
      _stream, _sm_gt_eq_80);

  if (_tw._use_gelu) {
    ker_bias_gelu_i8I_i8O_launcher<_DataType>(
        _batch_token_num, _stream, _int8_ffn_out_buf, _int8_ffn_in_buf,
        _p_device_wei[_weight_offset + 9], _tw._inner_size,
        _enc_clip_max[_layer_id * 11 + 10] / _quant_range,
        _quant_range / _enc_clip_max[_layer_id * 11 + 7], true, true);
  } else {
    ker_bias_relu_i8I_i8O_launcher<_DataType>(
        _batch_token_num, _stream, _int8_ffn_out_buf, _int8_ffn_in_buf,
        _p_device_wei[_weight_offset + 9], _tw._inner_size,
        _enc_clip_max[_layer_id * 11 + 10] / _quant_range,
        _quant_range / _enc_clip_max[_layer_id * 11 + 7],
        _enc_clip_max[_layer_id * 11 + 7], true, true, true);
  }

#ifdef DEBUG_RESULT
  for (int i = 0; i < _batch_size; i++) {       // batch_id
    for (int j = 0; j < _batch_seq_len; j++) {  // token_id
      std::cout << "ffn2 input: token-" << j << std::endl;
      print_vec(_int8_ffn_in_buf + i * _batch_seq_len * _tw._inner_size +
                    j * _tw._inner_size,
                "ffn2 input", 10);
    }
  }
#endif

  /* ---step 2. second ffn layer--- */
  cublasLtMM_withAlgo(_int32_ffn_out_buf, 1, _batch_token_num, _tw._hidden_size,
                      _tw._inner_size, 0, 0, 0, _int8_ffn_in_buf,
                      _int8_p_d_enc_wei[_layer_id * 4 + 3], _cublas_lt_handle,
                      _stream, _sm_gt_eq_80);

  const _DataType *scale_ptr, *bias_ptr, *res_bias_ptr;
  float clip_max, dequant_scale;
  if (_tw._use_gelu) {
    dequant_scale = _enc_clip_max[_layer_id * 11 + 3] *
                    _enc_clip_max[_layer_id * 11 + 7] /
                    (_quant_range * _quant_range);
  } else {
    dequant_scale = _enc_clip_max[_layer_id * 11 + 3] *
                    _enc_clip_max[_layer_id * 11 + 7] /
                    (2 * _quant_range * _quant_range);
  }
  if (_layer_id == _tw._n_enc_layer - 1) {
    scale_ptr = _p_d_src_emb_wei[4];
    bias_ptr = _p_d_src_emb_wei[5];

    ker_residual_bias_ln_i32I_launcher<_DataType>(
        _int32_ffn_out_buf, scale_ptr, bias_ptr, _p_d_output, _p_d_output,
        _batch_token_num, _tw._hidden_size, dequant_scale,
        _max_thread_per_block, _stream, true, _scaled_ffn2_colsum[_layer_id]);
  } else {
    scale_ptr = _p_device_wei[(_layer_id + 1) * _tw._weight_per_enc_layer];
    bias_ptr = _p_device_wei[(_layer_id + 1) * _tw._weight_per_enc_layer + 1];
    res_bias_ptr =
        _p_device_wei[(_layer_id + 1) * _tw._weight_per_enc_layer + 5];
    clip_max = _enc_clip_max[(_layer_id + 1) * 11 + 4];

    ker_residual_bias_ln_i32I_i8O_launcher<_DataType>(
        _int32_ffn_out_buf, scale_ptr, bias_ptr, res_bias_ptr, _int8_ffn_in_buf,
        _p_d_output, _batch_token_num, _tw._hidden_size, dequant_scale,
        _quant_range / clip_max, _max_thread_per_block, _stream,
        _tw._is_post_ln, true, true, _scaled_ffn2_colsum[_layer_id]);

#ifdef DEBUG_RESULT
    for (int i = 0; i < _batch_size; i++) {       // batch_id
      for (int j = 0; j < _batch_seq_len; j++) {  // token_id
        std::cout << "encoder layer out: token-" << j << std::endl;
        print_vec(_int8_ffn_in_buf + i * _batch_seq_len * _tw._hidden_size +
                      j * _tw._hidden_size,
                  "encoder layer out", 10);
      }
    }
#endif
  }

  return;
}

template class QuantVitEncoder<OperationType::FP16>;
template class QuantVitEncoder<OperationType::FP32>;

}  // namespace cuda
}  // namespace lightseq
