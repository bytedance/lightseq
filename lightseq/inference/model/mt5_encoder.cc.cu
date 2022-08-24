#include "mt5_encoder.h"

#include "../kernels/transformerKernels.h"
#include "../kernels/embKernels.h"
#include "../kernels/t5Kernels.h"
#include "../kernels/t5EmbKernels.h"

/**
@file
Transformer encoder, composed by gemm lib and
  custom cuda kernel function
*/

namespace lightseq {
namespace cuda {

template <OperationType OpType_>
MT5Encoder<OpType_>::MT5Encoder(int max_batch_size, int *p_d_token_id,
                                int *p_d_padding_mask, _DataType *p_d_output,
                                const MT5Weight<OpType_> &tw,
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

      _atten_scaler((_DataType)1.f),  // no atten_scaler
      _max_batch_dim(max_batch_size * tw._max_step * tw._hidden_size),
      _max_thread_per_block(1024) {}

/**
Compute GPU memory size needed by transformer encoder,
  to see how these memory is used, checkout init_buffer() for detail
*/
template <OperationType OpType_>
long MT5Encoder<OpType_>::compute_buffer_bytesize() {
  long sz1 = _max_batch_dim * 6 +
             _max_batch_size * _tw._head_num * _tw._max_step * _tw._max_step;
  long sz2 =
      _max_batch_dim + _max_batch_size * _tw._max_step * _tw._inner_size * 2;
  return max(sz1, sz2) * sizeof(_DataType);
}

/**
Init the GPU memory pointer which point to
  the memory buffer needed by encoder.
These buffer are used during custom cuda kernel function,
  find the corresponding function to see how these buffer are used
*/
template <OperationType OpType_>
void MT5Encoder<OpType_>::init_buffer(void *pbuf) {
  _DataType *p_d_buf = reinterpret_cast<_DataType *>(pbuf);
  _p_d_qkv_projected = p_d_buf;
  _p_d_q = _p_d_qkv_projected + _max_batch_dim * 3;
  _p_d_k = _p_d_q + _max_batch_dim;
  _p_d_v = _p_d_k + _max_batch_dim;
  _p_d_c = _p_d_v + _max_batch_dim;
  _p_d_ffn_buf1 = p_d_buf;
  _p_d_ffn_buf2 = _p_d_ffn_buf1 + _max_batch_dim;
  _p_d_ffn_buf3 =
      _p_d_ffn_buf2 + _max_batch_size * _tw._max_step * _tw._inner_size;
  // encoder and decoder use the same buffer to save gpu memory useage
  return;
}

/**
Some requirements needed by custom cuda kernel function
*/
template <OperationType OpType_>
std::string MT5Encoder<OpType_>::check() {
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
void MT5Encoder<OpType_>::run_one_infer(int batch_size, int batch_seq_len) {
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
  t5_launch_enc_emb<_DataType>(_p_d_src_emb_wei[0], _p_d_token_id, _p_d_output,
                               _p_d_padding_mask, _tw._padding_id, batch_size,
                               batch_seq_len, _tw._hidden_size, _stream,
                               _p_d_src_emb_wei[4], _p_d_lang_id);
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

  t5_ker_norm_layer_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_output, _p_d_output,
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
void MT5Encoder<OpType_>::self_attention() {
  /* ---step 0. layer_norm, add output_bias to "query"--- */

#ifdef DEBUG_RESULT
  printf("_weight_offset = %d\n", _weight_offset);
  print_vec(_p_d_enc_wei[0], "multihead_norm_scale weight", 10);
  print_vec(_p_d_enc_wei[1], "multihead_norm_bias weight", 10);
#endif

  t5_ker_norm_layer_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_output, _p_d_q,
      _p_d_enc_wei[_weight_offset], _p_d_enc_wei[_weight_offset + 1],
      _max_thread_per_block);

#ifdef DEBUG_RESULT
  for (int i = 0; i < _batch_size; i++) {       // batch_id
    for (int j = 0; j < _batch_seq_len; j++) {  // token_id
      std::cout << "after pre-norm: token-" << j << std::endl;
      print_vec(
          _p_d_q + i * _batch_seq_len * _tw._hidden_size + j * _tw._hidden_size,
          "hidden state: ", 10);
    }
  }
#endif

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

// printf _p_d_c
#ifdef DEBUG_RESULT
  std::cout << "encoder before softmax and position bias: " << std::endl;
  print_vec(_p_d_c, "_p_d_c matrix: ", 10);
#endif

  t5_ker_correlation_softmax_encself_launcher<_DataType>(
      _batch_size, _batch_seq_len, _tw._head_num, _stream, _p_d_c,
      _p_d_padding_mask, _p_d_src_emb_wei[1]);

#ifdef DEBUG_RESULT
  print_vec(_p_d_padding_mask, "_p_d_padding_mask: ", 2 * 22);
#endif
// printf _p_d_c
#ifdef DEBUG_RESULT
  std::cout << "after softmax: " << std::endl;
  print_vec(_p_d_c, "_p_d_c matrix: ", 22 * 22);
  print_vec(_p_d_c + 8 * 22 * 22, "_p_d_c matrix: ", 22 * 22);
#endif

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

#ifdef DEBUG_RESULT
  for (int i = 0; i < _batch_size; i++) {       // batch_id
    for (int j = 0; j < _batch_seq_len; j++) {  // token_id
      std::cout << "last of self-attention: token-" << j << std::endl;
      print_vec(_p_d_output + i * _batch_seq_len * _tw._hidden_size +
                    j * _tw._hidden_size,
                "hidden state: ", 10);
    }
  }
#endif

  return;
}

template <OperationType OpType_>
void MT5Encoder<OpType_>::ffn_add_norm() {
  /* ---step 0. layer_norm, add output_bias to "query"--- */

  t5_ker_norm_layer_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_output, _p_d_ffn_buf1,
      _p_d_enc_wei[_weight_offset + 6], _p_d_enc_wei[_weight_offset + 7],
      _max_thread_per_block);

#ifdef DEBUG_RESULT
  for (int i = 0; i < _batch_size; i++) {       // batch_id
    for (int j = 0; j < _batch_seq_len; j++) {  // token_id
      std::cout << "after ffn-pre-norm: token-" << j << std::endl;
      print_vec(_p_d_ffn_buf1 + i * _batch_seq_len * _tw._hidden_size +
                    j * _tw._hidden_size,
                "hidden state: ", 10);
    }
  }
  print_vec(_p_d_enc_wei[_weight_offset + 8], "ffn1_wei: ", 10);
#endif

  /* ---step 1. first ffn layer--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._inner_size, _batch_token_num,
      _tw._hidden_size, &_fone, _p_d_enc_wei[_weight_offset + 8], _AType,
      _tw._inner_size, _p_d_ffn_buf1, _BType, _tw._hidden_size, &_fzero,
      _p_d_ffn_buf2, _CType, _tw._inner_size, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

#ifdef DEBUG_RESULT
  std::cout << "result of first gemm" << std::endl;
  print_vec(_p_d_ffn_buf2, "result: ", 10);
#endif

  // if (_tw._use_gelu) {
  //   ker_bias_gelu_launcher<_DataType>(
  //       _batch_token_num, _max_thread_per_block, _stream, _p_d_ffn_buf2,
  //       _p_d_enc_wei[_weight_offset + 9], _tw._inner_size);
  // } else {
  //   ker_bias_relu_launcher<_DataType>(
  //       _batch_token_num, _max_thread_per_block, _stream, _p_d_ffn_buf2,
  //       _p_d_enc_wei[_weight_offset + 9], _tw._inner_size);
  // }

  // #ifdef DEBUG_RESULT
  //     std::cout << "result of first gemm(after gelu)" << std::endl;
  //     print_vec(_p_d_ffn_buf2, "result: ", 10);
  // #endif

  /* ---step 2. second ffn layer--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._inner_size, _batch_token_num,
      _tw._hidden_size, &_fone, _p_d_enc_wei[_weight_offset + 10], _AType,
      _tw._inner_size, _p_d_ffn_buf1, _BType, _tw._hidden_size, &_fzero,
      _p_d_ffn_buf3, _CType, _tw._inner_size, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

#ifdef DEBUG_RESULT
  std::cout << "result of second gemm" << std::endl;
  print_vec(_p_d_ffn_buf3, "result: ", 10);
#endif

  ker_gelu_first_elementmul_launcher<_DataType>(
      _batch_token_num, _max_thread_per_block, _stream, _p_d_ffn_buf2,
      _p_d_ffn_buf3, _tw._inner_size);

#ifdef DEBUG_RESULT
  std::cout << "result of gelu first and element wise multiply" << std::endl;
  print_vec(_p_d_ffn_buf2, "result: ", 10);
#endif

#ifdef DEBUG_RESULT
  std::cout << "weight of wo" << std::endl;
  print_vec(_p_d_enc_wei[_weight_offset + 12], "result: ", 10);
#endif

  /* ---step 3. third ffn layer--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size, _batch_token_num,
      _tw._inner_size, &_fone, _p_d_enc_wei[_weight_offset + 12], _AType,
      _tw._hidden_size, _p_d_ffn_buf2, _BType, _tw._inner_size, &_fone,
      _p_d_output, _CType, _tw._hidden_size, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

#ifdef DEBUG_RESULT
  for (int i = 0; i < _batch_size; i++) {       // batch_id
    for (int j = 0; j < _batch_seq_len; j++) {  // token_id
      std::cout << "last of ffn-layer: token-" << j << std::endl;
      print_vec(_p_d_output + i * _batch_seq_len * _tw._hidden_size +
                    j * _tw._hidden_size,
                "hidden state: ", 10);
    }
  }
#endif
  return;
}

template class MT5Encoder<OperationType::FP16>;
template class MT5Encoder<OperationType::FP32>;

}  // namespace cuda
}  // namespace lightseq
