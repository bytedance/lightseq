#include "src/custom/transformer/kernels/nmtKernels.h"
#include "src/custom/transformer/kernels/gptKernels.h"
#include "src/custom/transformer/model/gpt_encoder.h"

//#define DEBUG_RESULT
//#define DEBUG_TIME

namespace lab {
namespace nmt {

template <OperationType OpType_>
GptEncoder<OpType_>::GptEncoder(int max_batch_size, const int *p_d_token_id,
                          float *p_d_ppl,
                          const GptWeight<OpType_> &tw,
                          cudaStream_t stream, cublasHandle_t hd)
    : _max_batch_size(max_batch_size), _p_d_token_id(p_d_token_id),
      _p_d_ppl(p_d_ppl), _tw(tw),
      _stream(stream), _hd(hd), _p_d_src_emb_wei(tw.get_src_emb_wei()),
      _p_d_enc_wei(tw.get_enc_wei()), _fone((_DataType)1.f), _fzero((_DataType)0.f),
      _atten_scaler((_DataType)sqrt(1.f / tw._dim_per_head)),
      _max_batch_dim(max_batch_size * tw._max_step * tw._hidden_size),
      _max_thread_per_block(1024),
      _h_real_seq_len(max_batch_size, 0),
      _h_ppl(max_batch_size, 0.f) {}

template <OperationType OpType_>
int GptEncoder<OpType_>::compute_buffer_bytesize() {
  int si = _max_batch_size;
  int sz0 = _max_batch_dim;
  int sz1 = _max_batch_dim * 6 +
            _max_batch_size * _tw._head_num * _tw._max_step * _tw._max_step;
  int sz2 = _max_batch_dim + _max_batch_size * _tw._max_step * _tw._inner_size;
  int sz3 = _max_batch_size * _tw._max_step * _tw._src_vocab_size;
  return (sz0 + max(max(sz1, sz2), sz3)) * sizeof(_DataType) +
    si * sizeof(int);
}

template <OperationType OpType_>
void GptEncoder<OpType_>::init_buffer(void *pbuf) {

  // int buffer
  int *p_d_int = reinterpret_cast<int *>(pbuf);
  _p_d_real_seq_len = p_d_int;
  p_d_int += _max_batch_size;

  // datatype buffer
  _DataType *p_d_datatype = reinterpret_cast<_DataType *>(p_d_int);
  _p_d_query = p_d_datatype;
  p_d_datatype += _max_batch_dim;
  // reuse 1 ---------------------
  _p_d_qkv_projected = p_d_datatype;
  _p_d_q = _p_d_qkv_projected + _max_batch_dim * 3;
  _p_d_k = _p_d_q + _max_batch_dim;
  _p_d_v = _p_d_k + _max_batch_dim;
  // _max_batch_size * _tw._head_num *
  //  _tw._max_step * _tw._max_step
  _p_d_c = _p_d_v + _max_batch_dim;
  // reuse 2 ---------------------
  _p_d_ffn_buf1 = p_d_datatype;
  // _max_batch_size * _tw._max_step * _tw._inner_size
  _p_d_ffn_buf2 = _p_d_ffn_buf1 + _max_batch_dim; 
  // reuse 3 ---------------------
  // _max_batch_size * _tw._max_step * _tw._src_vocab_size
  _p_d_logit = p_d_datatype;
  return;
}

template <OperationType OpType_> std::string GptEncoder<OpType_>::check() {
  if (_max_thread_per_block < _tw._hidden_size) {
    return "violate hidden_size <= max_thread_per_block";
  }
  if (_tw._inner_size & 1) {
    return "violate inner_size % 2 = 0";
  }  
  if (_tw._dim_per_head & 1) {
    return "violate dim_per_head % 2 = 0";
  }
  if (_p_d_src_emb_wei.size() != 4) {
    return "violate p_d_src_emb_wei.size() = 4";
  }
  if (_p_d_enc_wei.size() != _tw._weight_per_enc_layer * _tw._n_enc_layer) {
    return "violate p_d_enc_wei.size() = weight_per_enc_layer * n_enc_layer";
  }
  return "";
}

template <OperationType OpType_>
void GptEncoder<OpType_>::run_one_infer(int batch_size, int batch_seq_len) {
  _batch_size = batch_size;
  _batch_seq_len = batch_seq_len;
  _batch_token_num = batch_size * batch_seq_len;
  CHECK_GPU_ERROR(cudaMemcpyAsync(_p_d_real_seq_len, _h_real_seq_len.data(),
             sizeof(int) * _batch_size,
             cudaMemcpyHostToDevice, _stream));
  CHECK_GPU_ERROR(cudaMemcpyAsync(_p_d_ppl, _h_ppl.data(),
             sizeof(float) * _batch_size,
             cudaMemcpyHostToDevice, _stream));

#ifdef DEBUG_RESULT
  std::cout << "batch_size-" << batch_size << " batch_seq_len-" << batch_seq_len << std::endl;
  print_vec(_p_d_token_id, "batch_token_ids", batch_size * batch_seq_len);
#endif
 
  // token embedding, add position embedding and layer_norm
  ker_gpt_embedding_launcher<_DataType>(
      batch_size, batch_seq_len, _tw._hidden_size, _stream, _p_d_src_emb_wei[0],
      _p_d_src_emb_wei[1], _p_d_token_id, _p_d_query, _p_d_real_seq_len,
      _tw._padding_id);

  for (_layer_id = 0; _layer_id < _tw._n_enc_layer; _layer_id++) {
    _weight_offset = _layer_id * _tw._weight_per_enc_layer;
    self_attention();
    ffn_add_norm();
  }

  ker_norm_layer_launcher<_DataType>(_batch_token_num, _tw._hidden_size, _stream,
      _p_d_query, _p_d_src_emb_wei[2], _p_d_src_emb_wei[3]);

  compute_ppl();

  return;
}

template <OperationType OpType_> void GptEncoder<OpType_>::self_attention() {
  // step 0. layer_norm
  ker_norm_layer_resual_launcher<_DataType>(_batch_token_num, _tw._hidden_size, _stream,
      _p_d_query, _p_d_q, _p_d_enc_wei[_weight_offset],
      _p_d_enc_wei[_weight_offset + 1], _p_d_enc_wei[_weight_offset + 5]);

  // step 1. qkv = q * qkv_wei + bias
  CHECK_GPU_ERROR(cublasGemmEx(_hd,
      CUBLAS_OP_N, CUBLAS_OP_N,
      _tw._hidden_size * 3, _batch_token_num, _tw._hidden_size,
      &_fone,
      _p_d_enc_wei[_weight_offset + 2], _AType, _tw._hidden_size * 3,
      _p_d_q, _BType, _tw._hidden_size,
      &_fzero,
      _p_d_qkv_projected, _CType, _tw._hidden_size * 3,
      _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  ker_arrange_encself_qkv_launcher<_DataType>(_batch_token_num, _tw._hidden_size, _stream,
      _p_d_qkv_projected, _p_d_enc_wei[_weight_offset + 3], _p_d_q,
      _max_batch_dim, _batch_seq_len, _tw._dim_per_head, _tw._head_num);

  // step 2. correlation = q * k
  CHECK_GPU_ERROR(cublasGemmStridedBatchedEx(_hd, 
    CUBLAS_OP_T, CUBLAS_OP_N, 
    _batch_seq_len, _batch_seq_len, _tw._dim_per_head, 
    &_atten_scaler, 
    _p_d_k, _AType, _tw._dim_per_head, _batch_seq_len * _tw._dim_per_head, 
    _p_d_q, _BType, _tw._dim_per_head, _batch_seq_len * _tw._dim_per_head, 
    &_fzero, 
    _p_d_c, _CType, _batch_seq_len, _batch_seq_len * _batch_seq_len,
    _batch_size * _tw._head_num, 
    _computeType, 
    CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  ker_correlation_softmax_gpt_launcher<_DataType>(_batch_size, 
      _batch_seq_len, _tw._head_num, _stream,
      _p_d_c, _p_d_real_seq_len);

  // step 3. q = correlation * v
  CHECK_GPU_ERROR(cublasGemmStridedBatchedEx(_hd,
      CUBLAS_OP_N, CUBLAS_OP_N,
      _tw._dim_per_head, _batch_seq_len, _batch_seq_len,
      &_fone,
      _p_d_v, _AType, _tw._dim_per_head, _batch_seq_len * _tw._dim_per_head,
      _p_d_c, _BType, _batch_seq_len, _batch_seq_len * _batch_seq_len,
      &_fzero,
      _p_d_q, _CType, _tw._dim_per_head, _batch_seq_len * _tw._dim_per_head,
      _batch_size * _tw._head_num,
      _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // use v to save reshaped q, since they are in same size and v
  // will not be use again before the next multi-head-attention
  ker_arrange_atten_output_launcher<_DataType>(_batch_token_num, 
      _tw._hidden_size, _stream, _p_d_q, _p_d_v, 
      _batch_seq_len, _tw._dim_per_head, _tw._head_num);

  // step 4. q = ori_q + q * output_wei
  CHECK_GPU_ERROR(cublasGemmEx(_hd,
    CUBLAS_OP_N, CUBLAS_OP_N,
    _tw._hidden_size, _batch_token_num, _tw._hidden_size,
    &_fone,
    _p_d_enc_wei[_weight_offset + 4], _AType, _tw._hidden_size,
    _p_d_v, _BType, _tw._hidden_size,
    &_fone,
    _p_d_query, _CType, _tw._hidden_size,
    _computeType,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  return;
}

template <OperationType OpType_> void GptEncoder<OpType_>::ffn_add_norm() {
  // step 0. layer_norm
  ker_norm_layer_resual_launcher<_DataType>(_batch_token_num, _tw._hidden_size, _stream,
      _p_d_query, _p_d_ffn_buf1, _p_d_enc_wei[_weight_offset + 6],
      _p_d_enc_wei[_weight_offset + 7], _p_d_enc_wei[_weight_offset + 11]);

  // step 1. first layer
  CHECK_GPU_ERROR(cublasGemmEx(_hd,
    CUBLAS_OP_N, CUBLAS_OP_N, 
    _tw._inner_size, _batch_token_num, _tw._hidden_size,
    &_fone,
    _p_d_enc_wei[_weight_offset + 8], _AType, _tw._inner_size,
    _p_d_ffn_buf1, _BType,  _tw._hidden_size,
    &_fzero,
    _p_d_ffn_buf2, _CType, _tw._inner_size,
    _computeType, 
    CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  ker_bias_gelu_launcher<_DataType>(_batch_token_num, _max_thread_per_block,
      _stream, _p_d_ffn_buf2, _p_d_enc_wei[_weight_offset + 9], _tw._inner_size);

  // step 2. second layer
  CHECK_GPU_ERROR(cublasGemmEx(_hd,
      CUBLAS_OP_N, CUBLAS_OP_N, 
      _tw._hidden_size, _batch_token_num, _tw._inner_size, 
      &_fone, 
      _p_d_enc_wei[_weight_offset + 10], _AType, _tw._hidden_size, 
      _p_d_ffn_buf2, _BType, _tw._inner_size, 
      &_fone,
      _p_d_query, _CType, _tw._hidden_size, 
      _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  return;
}

template <OperationType OpType_> void GptEncoder<OpType_>::compute_ppl() {  
  CHECK_GPU_ERROR(cublasGemmEx(
    _hd, 
    CUBLAS_OP_T, CUBLAS_OP_N, 
    _tw._src_vocab_size, _batch_token_num, _tw._hidden_size,
    &_fone,
    _p_d_src_emb_wei[0], _AType, _tw._hidden_size,
    _p_d_query, _BType, _tw._hidden_size,
    &_fzero,
    _p_d_logit, _CType, _tw._src_vocab_size,
    _computeType,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP));

#ifdef DEBUG_RESULT
  for(int i=0; i < _batch_size; i++) { // batch_id
      for(int j=0; j < 3; j++) { // token_id
        print_vec(_p_d_logit + i * _batch_seq_len * _tw._src_vocab_size + 
	    j * _tw._src_vocab_size, "logit", 3);
      }
  }
#endif

  ker_ppl_launcher<_DataType>(_batch_size, _batch_seq_len, 
    _max_thread_per_block, _stream, _p_d_logit, _p_d_token_id,
    _p_d_real_seq_len, _p_d_ppl, _tw._src_vocab_size);

}

template class GptEncoder<OperationType::FP16>;
template class GptEncoder<OperationType::FP32>;

}  // namespace nmt
}  // namespace lab
