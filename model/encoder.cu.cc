#include "src/custom/transformer/kernels/nmtKernels.h"
#include "src/custom/transformer/model/encoder.h"
#include "src/custom/transformer/util.h"

namespace lab {
namespace nmt {

Encoder::Encoder(int max_batch_size, const int* p_d_token_id,
                 int* p_d_padding_mask, float* p_d_output,
                 const TransformerWeight& tw, cudaStream_t stream, cublasHandle_t hd)
    : _max_batch_size(max_batch_size),
      _p_d_token_id(p_d_token_id),
      _p_d_padding_mask(p_d_padding_mask),
      _p_d_output(p_d_output),
      _tw(tw),
      _stream(stream), _hd(hd),
      _p_d_src_emb_wei(tw.get_src_emb_wei()),
      _p_d_enc_wei(tw.get_enc_wei()),
      _fone(1.f),
      _fzero(0.f),
      _atten_scaler(sqrt(1.f / tw._dim_per_head)),
      _max_batch_dim(max_batch_size * tw._max_step * tw._hidden_size),
      _max_thread_per_block(1024) {}

int Encoder::compute_buffer_bytesize() {
  int sz1 = _max_batch_dim * 6 +
            _max_batch_size * _tw._head_num * _tw._max_step * _tw._max_step;
  int sz2 = _max_batch_dim + _max_batch_size * _tw._max_step * _tw._inner_size;
  return max(sz1, sz2) * sizeof(float);
}

void Encoder::init_buffer(void* pbuf) {
  float* p_d_buf = reinterpret_cast<float*>(pbuf);
  _p_d_qkv_projected = p_d_buf;
  _p_d_q = _p_d_qkv_projected + _max_batch_dim * 3;
  _p_d_k = _p_d_q + _max_batch_dim;
  _p_d_v = _p_d_k + _max_batch_dim;
  _p_d_c = _p_d_v + _max_batch_dim;
  _p_d_ffn_buf1 = p_d_buf;
  _p_d_ffn_buf2 = _p_d_ffn_buf1 + _max_batch_dim;
  return;
}

std::string Encoder::check() {
  if (_max_thread_per_block < _tw._hidden_size) {
    return "violate hidden_size <= max_thread_per_block";
  }
  if (_tw._inner_size % _max_thread_per_block != 0 ||
      _tw._inner_size < _max_thread_per_block) {
    return "violate inner_size >= max_thread_per_block and inner_size % "
           "_max_thread_per_block = 0";
  }
  if (_p_d_src_emb_wei.size() != 4) {
    return "violate p_d_src_emb_wei.size() = 4";
  }
  if (_p_d_enc_wei.size() != _tw._weight_per_enc_layer * _tw._n_enc_layer) {
    return "violate p_d_enc_wei.size() = weight_per_enc_layer * n_enc_layer";
  }
  return "";
}

void Encoder::run_one_infer(int batch_size, int batch_seq_len) {
  _batch_size = batch_size;
  _batch_seq_len = batch_seq_len;
  _batch_token_num = batch_size * batch_seq_len;
  // token embedding, add position embedding and layer_norm
  ker_enc_embedding<<<dim3(batch_size, batch_seq_len), _tw._hidden_size, 0, _stream>>>(
      _p_d_src_emb_wei[0], _p_d_src_emb_wei[1], _p_d_token_id, _p_d_output,
      _p_d_padding_mask, _tw._padding_id);

  for (_layer_id = 0; _layer_id < _tw._n_enc_layer; _layer_id++) {
    _weight_offset = _layer_id * _tw._weight_per_enc_layer;
    self_attention();
    ffn_add_norm();
  }

  ker_norm_layer<<<_batch_token_num, _tw._hidden_size, 0, _stream>>>(
      _p_d_output, _p_d_src_emb_wei[2], _p_d_src_emb_wei[3]);
  return;
}

void Encoder::self_attention() {
  // step 0. layer_norm
  lab::nmt::ker_norm_layer3<<<_batch_token_num, _tw._hidden_size, 0, _stream>>>(
      _p_d_output, _p_d_q, _p_d_enc_wei[_weight_offset],
      _p_d_enc_wei[_weight_offset + 1], _p_d_enc_wei[_weight_offset + 5]);

  // step 1. qkv = q * qkv_wei + bias
  CUBLAS_CALL(cublasSgemm(_hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size * 3,
                          _batch_token_num, _tw._hidden_size, &_fone,
                          _p_d_enc_wei[_weight_offset + 2],
                          _tw._hidden_size * 3, _p_d_q, _tw._hidden_size,
                          &_fzero, _p_d_qkv_projected, _tw._hidden_size * 3));
  ker_arrange_encself_qkv<<<dim3(_batch_token_num, 3), _tw._hidden_size, 0, _stream>>>(
      _p_d_qkv_projected, _p_d_enc_wei[_weight_offset + 3], _p_d_q,
      _max_batch_dim, _batch_seq_len, _tw._dim_per_head, _tw._head_num);
  // step 2. correlation = q * k
  CUBLAS_CALL(cublasSgemmStridedBatched(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _batch_seq_len, _batch_seq_len,
      _tw._dim_per_head, &_atten_scaler, _p_d_k, _batch_seq_len,
      _batch_seq_len * _tw._dim_per_head, _p_d_q, _tw._dim_per_head,
      _batch_seq_len * _tw._dim_per_head, &_fzero, _p_d_c, _batch_seq_len,
      _batch_seq_len * _batch_seq_len, _batch_size * _tw._head_num));
  ker_correlation_softmax_encself<<<
      dim3(_batch_size, _tw._head_num * _batch_seq_len), _batch_seq_len, 0, _stream>>>(
      _p_d_c, _p_d_padding_mask);

  // step 3. q = correlation * v
  CUBLAS_CALL(cublasSgemmStridedBatched(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._dim_per_head, _batch_seq_len,
      _batch_seq_len, &_fone, _p_d_v, _tw._dim_per_head,
      _batch_seq_len * _tw._dim_per_head, _p_d_c, _batch_seq_len,
      _batch_seq_len * _batch_seq_len, &_fzero, _p_d_q, _tw._dim_per_head,
      _batch_seq_len * _tw._dim_per_head, _batch_size * _tw._head_num));

  // use v to save reshaped q, since they are in same size and v
  // will not be use again before the next multi-head-attention
  ker_arrange_atten_output<<<_batch_token_num, _tw._hidden_size, 0, _stream>>>(
      _p_d_q, _p_d_v, _batch_seq_len, _tw._dim_per_head, _tw._head_num);

  // step 4. q = ori_q + q * output_wei
  CUBLAS_CALL(cublasSgemm(_hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size,
                          _batch_token_num, _tw._hidden_size, &_fone,
                          _p_d_enc_wei[_weight_offset + 4], _tw._hidden_size,
                          _p_d_v, _tw._hidden_size, &_fone, _p_d_output,
                          _tw._hidden_size));

  return;
}

void Encoder::ffn_add_norm() {
  // step 0. layer_norm
  lab::nmt::ker_norm_layer3<<<_batch_token_num, _tw._hidden_size, 0, _stream>>>(
      _p_d_output, _p_d_ffn_buf1, _p_d_enc_wei[_weight_offset + 6],
      _p_d_enc_wei[_weight_offset + 7], _p_d_enc_wei[_weight_offset + 11]);

  // step 1. first layer
  CUBLAS_CALL(cublasSgemm(_hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._inner_size,
                          _batch_token_num, _tw._hidden_size, &_fone,
                          _p_d_enc_wei[_weight_offset + 8], _tw._inner_size,
                          _p_d_ffn_buf1, _tw._hidden_size, &_fzero,
                          _p_d_ffn_buf2, _tw._inner_size));
  kerBiasRelu<<<dim3(_batch_token_num, _tw._inner_size / _max_thread_per_block),
                _max_thread_per_block, 0, _stream>>>(
      _p_d_ffn_buf2, _p_d_enc_wei[_weight_offset + 9], _tw._inner_size);
  // step 2. second layer
  CUBLAS_CALL(cublasSgemm(_hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size,
                          _batch_token_num, _tw._inner_size, &_fone,
                          _p_d_enc_wei[_weight_offset + 10], _tw._hidden_size,
                          _p_d_ffn_buf2, _tw._inner_size, &_fone, _p_d_output,
                          _tw._hidden_size));
  // kerBiasAdd<<<_batch_token_num, _tw._hidden_size>>>(_p_d_output,
  //        _p_d_ffn_buf1, _p_d_enc_wei[_weight_offset+11]);

  return;
}

}  // namespace nmt
}  // namespace lab
