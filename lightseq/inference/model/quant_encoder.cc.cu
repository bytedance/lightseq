#include "quant_encoder.h"

#include "../kernels/transformerKernels.h"
#include "../kernels/embKernels_int8.h"
#include "../kernels/transformerKernels_int8.h"
#include "cublas_helper.h"

/**
@file
QuantTransformer encoder, composed by gemm lib and
  custom cuda kernel function
*/

namespace lightseq {
namespace cuda {

template <OperationType OpType_>
QuantEncoder<OpType_>::QuantEncoder(int max_batch_size, int *p_d_token_id,
                                    int *p_d_padding_mask,
                                    _DataType *p_d_output,
                                    const QuantTransformerWeight<OpType_> &tw,
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

      _src_emb_clip_max(tw.get_src_emb_clip_max()),
      _enc_clip_max(tw.get_enc_clip_max()),
      _ione((int32_t)1),
      _izero((int32_t)0),

      _atten_scaler((_DataType)sqrt(1.f / tw._dim_per_head)),
      _max_batch_dim(max_batch_size * tw._max_step * tw._hidden_size),
      _max_thread_per_block(1024),
      _algo_map(),
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
void QuantEncoder<OpType_>::init_buffer() {
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

  CHECK_GPU_ERROR(
      cudaMalloc(&_int8_p_d_src_emb_wei,
                 _tw._src_vocab_size * _tw._hidden_size * sizeof(int8_t)));
  quantize_weight(_p_d_src_emb_wei[0], _int8_p_d_src_emb_wei,
                  _tw._src_vocab_size, _tw._hidden_size,
                  _quant_range / _src_emb_clip_max, _stream, _cublas_lt_handle,
                  kRowMajor);

  _p_device_emb.push_back(nullptr);
  _p_device_emb.push_back(
      to_gpu(_p_d_src_emb_wei[1], _tw._max_step * _tw._hidden_size, _stream));
  _p_device_emb.push_back(
      to_gpu(_p_d_src_emb_wei[2], _tw._hidden_size, _stream));
  _p_device_emb.push_back(
      to_gpu(_p_d_src_emb_wei[3], _tw._hidden_size, _stream));
  if (_tw._multilg_type != 0) {
    _p_device_emb.push_back(
        to_gpu(_p_d_src_emb_wei[4], _tw._hidden_size, _stream));
  } else {
    _p_device_emb.push_back(nullptr);
  }

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

    auto weight_layout = _sm_gt_eq_80 ? kColMajor : kColMajor32;

    quantize_weight(_p_d_enc_wei[_weight_offset + 2],
                    _int8_p_d_enc_wei[_layer_id * 4], _tw._hidden_size,
                    _tw._hidden_size * 3,
                    _quant_range / _enc_clip_max[_layer_id * 12], _stream,
                    _cublas_lt_handle, weight_layout);

    quantize_weight(_p_d_enc_wei[_weight_offset + 4],
                    _int8_p_d_enc_wei[_layer_id * 4 + 1], _tw._hidden_size,
                    _tw._hidden_size,
                    _quant_range / _enc_clip_max[_layer_id * 12 + 1], _stream,
                    _cublas_lt_handle, weight_layout);

    quantize_weight(_p_d_enc_wei[_weight_offset + 8],
                    _int8_p_d_enc_wei[_layer_id * 4 + 2], _tw._hidden_size,
                    _tw._inner_size,
                    _quant_range / _enc_clip_max[_layer_id * 12 + 2], _stream,
                    _cublas_lt_handle, weight_layout);

    quantize_weight(_p_d_enc_wei[_weight_offset + 10],
                    _int8_p_d_enc_wei[_layer_id * 4 + 3], _tw._inner_size,
                    _tw._hidden_size,
                    _quant_range / _enc_clip_max[_layer_id * 12 + 3], _stream,
                    _cublas_lt_handle, weight_layout);

    if (_tw._use_gelu) {
      _scaled_ffn2_colsum[_layer_id] = nullptr;
    } else {
      CHECK_GPU_ERROR(cudaMalloc(&_scaled_ffn2_colsum[_layer_id],
                                 _tw._hidden_size * sizeof(_DataType)));
      float relu_scale = _enc_clip_max[_layer_id * 12 + 7] / 2;
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
std::string QuantEncoder<OpType_>::check() {
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
QuantEncoder inference
*/
template <OperationType OpType_>
void QuantEncoder<OpType_>::run_one_infer(int batch_size, int batch_seq_len) {
  if (batch_size > _max_batch_size) {
    throw std::runtime_error("batch size of input greater than max_batch_size");
  }
  if (batch_seq_len > _tw._max_step) {
    throw std::runtime_error("seq len of input greater than max_step");
  }

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
  launch_enc_emb_i8I<_DataType>(
      _int8_p_d_src_emb_wei, _p_device_emb[1], _p_d_token_id, _p_d_output,
      _p_d_padding_mask, _tw._padding_id, batch_size, batch_seq_len,
      _tw._hidden_size, _stream, _p_device_emb[4], _p_d_lang_id,
      _tw._multilg_type, _src_emb_clip_max / _quant_range, true);
#ifdef DEBUG_RESULT
  for (int i = 0; i < _batch_size; i++) {       // batch_id
    for (int j = 0; j < _batch_seq_len; j++) {  // token_id
      std::cout << "emb out: token-" << j << std::endl;
      print_vec(_p_d_output + i * _batch_seq_len * _tw._hidden_size +
                    j * _tw._hidden_size,
                "emb out", 10);
    }
  }  // not normal
  print_vec(_int8_p_d_src_emb_wei, "token embedding weight", 10);
  print_vec(_p_device_emb[1], "position embedding weight", 10);
#endif
  for (_layer_id = 0; _layer_id < _tw._n_enc_layer; _layer_id++) {
    _weight_offset = _layer_id * _tw._weight_per_enc_layer;
    self_attention();
    ffn_add_norm();
  }

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
QuantEncoder self attention
*/
template <OperationType OpType_>
void QuantEncoder<OpType_>::self_attention() {
  if (_layer_id == 0) {
    ker_norm_layer_resual_i8O_launcher<_DataType>(
        _batch_token_num, _tw._hidden_size, _stream, _p_d_output,
        _int8_ffn_in_buf, _p_device_wei[_weight_offset],
        _p_device_wei[_weight_offset + 1], _p_device_wei[_weight_offset + 5],
        _max_thread_per_block, _quant_range / _enc_clip_max[_layer_id * 12 + 4],
        _tw._is_post_ln, !_sm_gt_eq_80);
  }
  CHECK_GPU_ERROR(cudaGetLastError());

  if (_sm_gt_eq_80) {
    cublaslt_gemm(
        _int8_p_d_enc_wei[_layer_id * 4], _int8_ffn_in_buf, _int8_ffn_out_buf,
        1, _tw._hidden_size * 3, _batch_token_num, _tw._hidden_size, 0, 0, 0,
        _enc_clip_max[_layer_id * 12] * _enc_clip_max[_layer_id * 12 + 4] /
            (_enc_clip_max[_layer_id * 12 + 8] * _quant_range),
        _cublas_lt_handle, _stream, _algo_map);
  } else {
    cublasLtMM_withAlgo_i8IO(
        _int8_ffn_out_buf, 1, _batch_token_num, _tw._hidden_size * 3,
        _tw._hidden_size, 0, 0, 0,
        _enc_clip_max[_layer_id * 12] * _enc_clip_max[_layer_id * 12 + 4] /
            (_enc_clip_max[_layer_id * 12 + 8] * _quant_range),
        _int8_ffn_in_buf, _int8_p_d_enc_wei[_layer_id * 4], _cublas_lt_handle,
        _stream, _sm_gt_eq_80);
  }

  // get q, k, v by split and reshape qkv

  ker_arrange_encself_qkv_i8I_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _int8_ffn_out_buf,
      _p_device_wei[_weight_offset + 3], _p_d_q, _max_batch_dim, _batch_seq_len,
      _tw._dim_per_head, _tw._head_num, _max_thread_per_block,
      _enc_clip_max[_layer_id * 12 + 8] / _quant_range, !_sm_gt_eq_80);

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
      _quant_range / _enc_clip_max[_layer_id * 12 + 5], !_sm_gt_eq_80);

  /* ---step 4. new_q = ori_q + new_q * output_wei--- */

  if (_sm_gt_eq_80) {
    cublaslt_gemm(_int8_p_d_enc_wei[_layer_id * 4 + 1], _int8_ffn_in_buf,
                  _int8_ffn_out_buf, 1, _tw._hidden_size, _batch_token_num,
                  _tw._hidden_size, 0, 0, 0,
                  _enc_clip_max[_layer_id * 12 + 1] *
                      _enc_clip_max[_layer_id * 12 + 5] /
                      (_enc_clip_max[_layer_id * 12 + 9] * _quant_range),
                  _cublas_lt_handle, _stream, _algo_map);
  } else {
    cublasLtMM_withAlgo_i8IO(
        _int8_ffn_out_buf, 1, _batch_token_num, _tw._hidden_size,
        _tw._hidden_size, 0, 0, 0,
        _enc_clip_max[_layer_id * 12 + 1] * _enc_clip_max[_layer_id * 12 + 5] /
            (_enc_clip_max[_layer_id * 12 + 9] * _quant_range),
        _int8_ffn_in_buf, _int8_p_d_enc_wei[_layer_id * 4 + 1],
        _cublas_lt_handle, _stream, _sm_gt_eq_80);
  }

  ker_residual_bias_ln_i8I_i8O_launcher<_DataType>(
      _int8_ffn_out_buf, _p_device_wei[_weight_offset + 6],
      _p_device_wei[_weight_offset + 7], _p_device_wei[_weight_offset + 11],
      _int8_ffn_in_buf, _p_d_output, _batch_token_num, _tw._hidden_size,
      _enc_clip_max[_layer_id * 12 + 9] / _quant_range,
      _quant_range / _enc_clip_max[_layer_id * 12 + 6], _max_thread_per_block,
      _stream, _tw._is_post_ln, !_sm_gt_eq_80, !_sm_gt_eq_80);

  return;
}

template <OperationType OpType_>
void QuantEncoder<OpType_>::ffn_add_norm() {
  if (_sm_gt_eq_80) {
    cublaslt_gemm(_int8_p_d_enc_wei[_layer_id * 4 + 2], _int8_ffn_in_buf,
                  _int8_ffn_out_buf, 1, _tw._inner_size, _batch_token_num,
                  _tw._hidden_size, 0, 0, 0,
                  _enc_clip_max[_layer_id * 12 + 2] *
                      _enc_clip_max[_layer_id * 12 + 6] /
                      (_enc_clip_max[_layer_id * 12 + 10] * _quant_range),
                  _cublas_lt_handle, _stream, _algo_map);
  } else {
    cublasLtMM_withAlgo_i8IO(
        _int8_ffn_out_buf, 1, _batch_token_num, _tw._inner_size,
        _tw._hidden_size, 0, 0, 0,
        _enc_clip_max[_layer_id * 12 + 2] * _enc_clip_max[_layer_id * 12 + 6] /
            (_enc_clip_max[_layer_id * 12 + 10] * _quant_range),
        _int8_ffn_in_buf, _int8_p_d_enc_wei[_layer_id * 4 + 2],
        _cublas_lt_handle, _stream, _sm_gt_eq_80);
  }

  if (_tw._use_gelu) {
    ker_bias_gelu_i8I_i8O_launcher<_DataType>(
        _batch_token_num, _stream, _int8_ffn_out_buf, _int8_ffn_in_buf,
        _p_device_wei[_weight_offset + 9], _tw._inner_size,
        _enc_clip_max[_layer_id * 12 + 10] / _quant_range,
        _quant_range / _enc_clip_max[_layer_id * 12 + 7], !_sm_gt_eq_80,
        !_sm_gt_eq_80);
  } else {
    ker_bias_relu_i8I_i8O_launcher<_DataType>(
        _batch_token_num, _stream, _int8_ffn_out_buf, _int8_ffn_in_buf,
        _p_device_wei[_weight_offset + 9], _tw._inner_size,
        _enc_clip_max[_layer_id * 12 + 10] / _quant_range,
        _quant_range / _enc_clip_max[_layer_id * 12 + 7],
        _enc_clip_max[_layer_id * 12 + 7], !_sm_gt_eq_80, !_sm_gt_eq_80, true);
  }

  /* ---step 2. second ffn layer--- */
  if (_sm_gt_eq_80) {
    cublaslt_gemm(_int8_p_d_enc_wei[_layer_id * 4 + 3], _int8_ffn_in_buf,
                  _int32_ffn_out_buf, 1, _tw._hidden_size, _batch_token_num,
                  _tw._inner_size, 0, 0, 0, 1, _cublas_lt_handle, _stream,
                  _algo_map);
  } else {
    cublasLtMM_withAlgo(_int32_ffn_out_buf, 1, _batch_token_num,
                        _tw._hidden_size, _tw._inner_size, 0, 0, 0,
                        _int8_ffn_in_buf, _int8_p_d_enc_wei[_layer_id * 4 + 3],
                        _cublas_lt_handle, _stream, _sm_gt_eq_80);
  }

  const _DataType *scale_ptr, *bias_ptr, *res_bias_ptr;
  float clip_max, dequant_scale;
  if (_tw._use_gelu) {
    dequant_scale = _enc_clip_max[_layer_id * 12 + 3] *
                    _enc_clip_max[_layer_id * 12 + 7] /
                    (_quant_range * _quant_range);
  } else {
    dequant_scale = _enc_clip_max[_layer_id * 12 + 3] *
                    _enc_clip_max[_layer_id * 12 + 7] /
                    (2 * _quant_range * _quant_range);
  }
  if (_layer_id == _tw._n_enc_layer - 1) {
    scale_ptr = _p_device_emb[2];
    bias_ptr = _p_device_emb[3];

    ker_residual_bias_ln_i32I_launcher<_DataType>(
        _int32_ffn_out_buf, scale_ptr, bias_ptr, _p_d_output, _p_d_output,
        _batch_token_num, _tw._hidden_size, dequant_scale,
        _max_thread_per_block, _stream, !_sm_gt_eq_80,
        _scaled_ffn2_colsum[_layer_id]);
  } else {
    scale_ptr = _p_device_wei[(_layer_id + 1) * _tw._weight_per_enc_layer];
    bias_ptr = _p_device_wei[(_layer_id + 1) * _tw._weight_per_enc_layer + 1];
    res_bias_ptr =
        _p_device_wei[(_layer_id + 1) * _tw._weight_per_enc_layer + 5];
    clip_max = _enc_clip_max[(_layer_id + 1) * 12 + 4];

    ker_residual_bias_ln_i32I_i8O_launcher<_DataType>(
        _int32_ffn_out_buf, scale_ptr, bias_ptr, res_bias_ptr, _int8_ffn_in_buf,
        _p_d_output, _batch_token_num, _tw._hidden_size, dequant_scale,
        _quant_range / clip_max, _max_thread_per_block, _stream,
        _tw._is_post_ln, !_sm_gt_eq_80, !_sm_gt_eq_80,
        _scaled_ffn2_colsum[_layer_id]);
  }

  return;
}

template class QuantEncoder<OperationType::FP16>;
template class QuantEncoder<OperationType::FP32>;

}  // namespace cuda
}  // namespace lightseq
