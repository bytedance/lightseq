#include "moe_encoder.h"
#include "../kernels/transformerKernels.h"
#include "../kernels/embKernels.h"
#include "../kernels/moeKernels.h"

/**
@file
MoE encoder, composed by gemm lib and
  custom cuda kernel function
*/

namespace lightseq {
namespace cuda {

template <OperationType OpType_>
MoeEncoder<OpType_>::MoeEncoder(int max_batch_size, int *p_d_token_id,
                                int *p_d_padding_mask, _DataType *p_d_output,
                                const MoeWeight<OpType_> &tw,
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
      _atten_scaler((_DataType)sqrt(1.f / tw._dim_per_head)),
      _max_batch_dim(max_batch_size * tw._max_step * tw._hidden_size),
      _max_token_num(max_batch_size * tw._max_step),
      _max_thread_per_block(1024),
      _gate_weight_offset(0),
      _p_d_enc_gate_wei(tw.get_enc_gate_wei()) {}

/**
Compute GPU memory size needed by moe_encoder,
  to see how these memory is used, checkout init_buffer() for detail
*/
template <OperationType OpType_>
long MoeEncoder<OpType_>::compute_buffer_bytesize() {
  long sz1 = _max_batch_dim * 6 +
             _max_batch_size * _tw._head_num * _tw._max_step * _tw._max_step;
  long sz2 = _max_batch_dim + _max_token_num * _tw._inner_size;
  long sz3 = _max_token_num * _tw._expert_num_encoder *
                 (_max_batch_dim + _max_token_num * _tw._inner_size + 1) *
                 sizeof(_DataType) +
             _max_token_num * _tw._expert_num_encoder * sizeof(float) +
             _tw._moe_topk_encoder * _max_token_num * sizeof(int);

  return max(sz1 * sizeof(_DataType), sz2 * sizeof(_DataType) + sz3);
}

/**
Init the GPU memory pointer which point to
  the memory buffer needed by moe_encoder.
These buffer are used during custom cuda kernel function,
  find the corresponding function to see how these buffer are used
*/
template <OperationType OpType_>
void MoeEncoder<OpType_>::init_buffer(void *pbuf) {
  _DataType *p_d_buf = reinterpret_cast<_DataType *>(pbuf);
  _p_d_qkv_projected = p_d_buf;
  _p_d_q = _p_d_qkv_projected + _max_batch_dim * 3;
  _p_d_k = _p_d_q + _max_batch_dim;
  _p_d_v = _p_d_k + _max_batch_dim;
  _p_d_c = _p_d_v + _max_batch_dim;
  _p_d_ffn_buf1 = p_d_buf;
  _p_d_ffn_buf2 = _p_d_ffn_buf1 + _max_batch_dim;
  _p_d_gate = _p_d_ffn_buf2 + _max_token_num * _tw._inner_size;
  _p_d_score_routed = reinterpret_cast<float *>(
      _p_d_gate + _max_token_num * _tw._expert_num_encoder);
  _p_d_expert_id_routed = reinterpret_cast<int *>(
      _p_d_score_routed + _max_token_num * _tw._expert_num_encoder);
  _p_d_moe_input_buf = reinterpret_cast<_DataType *>(
      _p_d_expert_id_routed + _tw._moe_topk_encoder * _max_token_num);
  _p_d_moe_inner_buf =
      _p_d_moe_input_buf + _tw._expert_num_encoder * _max_batch_dim;
  // encoder and decoder use the same buffer to save gpu memory useage

  return;
}

/**
Some requirements needed by custom cuda kernel function
*/
template <OperationType OpType_>
std::string MoeEncoder<OpType_>::check() {
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
  if (_tw._moe_topk_encoder != 1 && _tw._moe_topk_encoder != 2) {
    return "moe topk should be 1 or 2";
  }
  if (_tw._expert_num_encoder > 1024) {
    return "number of moe expert should not be greater than 1024";
  }
  return "";
}

/**
Encoder inference
*/
template <OperationType OpType_>
void MoeEncoder<OpType_>::run_one_infer(int batch_size, int batch_seq_len) {
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
  _gate_weight_offset = 0;
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
  print_vec(_p_d_src_emb_wei[0], "token embedding weight", 10);
  print_vec(_p_d_src_emb_wei[1], "position embedding weight", 10);
#endif
  for (_layer_id = 0; _layer_id < _tw._n_enc_layer; _layer_id++) {
    _weight_offset = _layer_id * _tw._weight_per_enc_layer;
    self_attention();
    ffn_add_norm();
  }
  // last layer norm
  ker_norm_layer_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_output,
      _p_d_src_emb_wei[2], _p_d_src_emb_wei[3], _max_thread_per_block);

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
void MoeEncoder<OpType_>::self_attention() {
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
  ker_arrange_atten_output_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_q, _p_d_v,
      _batch_seq_len, _tw._dim_per_head, _tw._head_num, _max_thread_per_block);

  /* ---step 4. new_q = ori_q + new_q * output_wei--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size, _batch_token_num,
      _tw._hidden_size, &_fone, _p_d_enc_wei[_weight_offset + 4], _AType,
      _tw._hidden_size, _p_d_v, _BType, _tw._hidden_size, &_fone, _p_d_output,
      _CType, _tw._hidden_size, _computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  return;
}

template <OperationType OpType_>
void MoeEncoder<OpType_>::ffn_add_norm() {
  if (_tw._is_moe_layer_encoder[_layer_id]) {
    moe_fw();
    ++_gate_weight_offset;
  } else {
    ffn();
  }
  return;
}

template <OperationType OpType_>
void MoeEncoder<OpType_>::ffn() {
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
  return;
}

template <OperationType OpType_>
void MoeEncoder<OpType_>::moe_fw() {
  ker_norm_layer_prepost_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_output, _p_d_ffn_buf1,
      _p_d_enc_wei[_weight_offset + 6], _p_d_enc_wei[_weight_offset + 7],
      _max_thread_per_block, _tw._is_post_ln);

  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._expert_num_encoder, _batch_token_num,
      _tw._hidden_size, &_fone, _p_d_enc_gate_wei[_gate_weight_offset], _AType,
      _tw._expert_num_encoder, _p_d_ffn_buf1, _BType, _tw._hidden_size, &_fzero,
      _p_d_gate, _CType, _tw._expert_num_encoder, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  ker_softmax_topk_router_launcher<_DataType>(
      _batch_token_num, _tw._expert_num_encoder, _max_token_num,
      _tw._moe_topk_encoder, _stream, _p_d_gate, _p_d_score_routed,
      _p_d_expert_id_routed);

  ker_reorder_tokens_launcher<_DataType>(
      _batch_token_num, _tw._expert_num_encoder, _max_token_num,
      _tw._hidden_size, _max_thread_per_block, _stream, _p_d_ffn_buf1,
      _p_d_score_routed, _p_d_moe_input_buf);

  CHECK_GPU_ERROR(cublasGemmStridedBatchedEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._inner_size, _batch_token_num,
      _tw._hidden_size, &_fone, _p_d_enc_wei[_weight_offset + 8], _AType,
      _tw._inner_size, _tw._hidden_size * _tw._inner_size, _p_d_moe_input_buf,
      _BType, _tw._hidden_size, _tw._hidden_size * _max_token_num, &_fzero,
      _p_d_moe_inner_buf, _CType, _tw._inner_size,
      _tw._inner_size * _max_token_num, _tw._expert_num_encoder, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  if (_tw._use_gelu) {
    ker_strided_bias_gelu_launcher<_DataType>(
        _batch_token_num, _tw._expert_num_encoder, _max_token_num,
        _tw._inner_size, _max_thread_per_block, _stream, _p_d_moe_inner_buf,
        _p_d_enc_wei[_weight_offset + 9]);
  } else {
    ker_strided_bias_relu_launcher<_DataType>(
        _batch_token_num, _tw._expert_num_encoder, _max_token_num,
        _tw._inner_size, _max_thread_per_block, _stream, _p_d_moe_inner_buf,
        _p_d_enc_wei[_weight_offset + 9]);
  }

  CHECK_GPU_ERROR(cublasGemmStridedBatchedEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size, _batch_token_num,
      _tw._inner_size, &_fone, _p_d_enc_wei[_weight_offset + 10], _AType,
      _tw._hidden_size, _tw._hidden_size * _tw._inner_size, _p_d_moe_inner_buf,
      _BType, _tw._inner_size, _tw._inner_size * _max_token_num, &_fzero,
      _p_d_moe_input_buf, _CType, _tw._hidden_size,
      _tw._hidden_size * _max_token_num, _tw._expert_num_encoder, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  ker_bias_redirect_residual_launcher<_DataType>(
      _tw._hidden_size, _max_token_num, _tw._moe_topk_encoder, _batch_token_num,
      _max_thread_per_block, _stream, _p_d_moe_input_buf,
      _p_d_enc_wei[_weight_offset + 11], _p_d_score_routed,
      _p_d_expert_id_routed, _p_d_output);
}

template class MoeEncoder<OperationType::FP16>;
template class MoeEncoder<OperationType::FP32>;

}  // namespace cuda
}  // namespace lightseq
