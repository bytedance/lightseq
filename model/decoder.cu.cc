#include <cub/cub.cuh>

#include "src/custom/transformer/kernels/nmtKernels.h"
#include "src/custom/transformer/model/decoder.h"
#include "src/custom/transformer/util.h"

//#define DEBUG_RESULT
//#define DEBUG_TIME

namespace lab {
namespace nmt {

Decoder::Decoder(int max_batch_size, const int* p_d_padding_mask,
                 const float* p_d_encoder_output, int* p_d_result,
                 const TransformerWeight& tw, cublasHandle_t hd)
    : _max_batch_size(max_batch_size),
      _max_thread_per_block(1024),
      _h_can_num_batch(0),
      _cub_sort_buffer_bytes(max_batch_size * tw._beam_size *
                             tw._trg_vocab_size * sizeof(float)),
      _p_d_padding_mask(p_d_padding_mask),
      _p_d_encoder_output(p_d_encoder_output),
      _p_d_result(p_d_result),
      _p_d_trg_emb_wei(tw.get_trg_emb_wei()),
      _p_d_dec_wei(tw.get_dec_wei()),
      _tw(tw),
      _hd(hd),
      _layer_size_encdec_k(max_batch_size * tw._max_step * tw._hidden_size),
      _layer_size_self_k(max_batch_size * tw._max_step * tw._hidden_size *
                         tw._beam_size),
      _fone(1.f),
      _fzero(0.f),
      _atten_scaler(sqrt(1.f / tw._dim_per_head)),
      _output_scaler(sqrt(1.f / tw._hidden_size)),
      _h_alive_seq_probs(max_batch_size * tw._beam_size,
                         min_log_probability / 2),
      _h_length_norm(tw._max_step, 1.f) {
  for (int i = 0; i < _h_alive_seq_probs.size(); i += tw._beam_size) {
    _h_alive_seq_probs[i] = 0.f;
  }
  if (tw._length_penalty >= 0) {
    for (int i = 0; i < _h_length_norm.size(); i++) {
      _h_length_norm[i] = length_norm(i + 1, tw._length_penalty);
    }
  }
#ifdef DEBUG_RESULT
  std::cout << "length penalty is: " << tw._length_penalty << std::endl;
  std::cout << "length norm is: {";
  for (int i = 0; i < _h_length_norm.size(); i++) {
    std::cout << _h_length_norm[i] << " ";
  }
  std::cout << "}" << std::endl;
#endif
  return;
}

int Decoder::compute_buffer_bytesize() {
  // compute buffer size, init_buffer() shows how the buffer is used
  int cache_bytesize = 4 * _tw._n_dec_layer * _layer_size_self_k +
                       2 * _tw._n_dec_layer * _layer_size_encdec_k +
                       _max_batch_size * _tw._beam_size * _tw._hidden_size;
  cache_bytesize *= sizeof(float);

  int decode_buffer_bytesize =
      _max_batch_size * _tw._beam_size * _tw._hidden_size * 4 +
      _max_batch_size * _tw._beam_size *
          max(_tw._hidden_size, _tw._inner_size) +
      _max_batch_size * _tw._head_num * _tw._beam_size * _tw._max_step;
  decode_buffer_bytesize *= sizeof(float);

  int sf = _max_batch_size * _tw._beam_size * _tw._trg_vocab_size * 2 +
           _max_batch_size * _tw._beam_size * 2;
  int si = _max_batch_size * _tw._beam_size * _tw._max_step * 2 +
           _max_batch_size * _tw._beam_size * _tw._trg_vocab_size +
           _max_batch_size * _tw._beam_size + 1;
  int beam_buffer_bytesize = sf * sizeof(float) + si * sizeof(int);

  return cache_bytesize + max(decode_buffer_bytesize, beam_buffer_bytesize);
}

void Decoder::init_buffer(void* pbuf) {
  float* curp = reinterpret_cast<float*>(pbuf);

  for (int i = 0; i < _tw._n_dec_layer; i++) {
    _p_d_encdec_k_bgeem.push_back(curp);
    curp += _layer_size_encdec_k;
  }
  for (int i = 0; i < _tw._n_dec_layer; i++) {
    _p_d_encdec_v_bgeem.push_back(curp);
    curp += _layer_size_encdec_k;
  }
  // reused buffer with _p_d_self_k_bgeem _p_d_self_v_bgeem,
  // no need to add curp because _p_d_encoder_out_buf is smaller
  // and no need to use it any more after get _p_d_encdec_k_bgeem
  // and _p_d_encdec_v_bgeem
  _p_d_encoder_out_buf = curp;

  for (int i = 0; i < _tw._n_dec_layer * 2; i++) {
    _p_d_self_k_bgeem.push_back(curp);
    curp += _layer_size_self_k;
  }
  for (int i = 0; i < _tw._n_dec_layer * 2; i++) {
    _p_d_self_v_bgeem.push_back(curp);
    curp += _layer_size_self_k;
  }
  _p_d_self_k_bgeem1 = _p_d_self_k_bgeem.data();
  _p_d_self_k_bgeem2 = _p_d_self_k_bgeem.data() + _tw._n_dec_layer;
  _p_d_self_v_bgeem1 = _p_d_self_v_bgeem.data();
  _p_d_self_v_bgeem2 = _p_d_self_v_bgeem.data() + _tw._n_dec_layer;

  _p_d_cur_step_query = curp;
  curp += _max_batch_size * _tw._beam_size * _tw._hidden_size;

  float* reuse_p = curp;

  // for decode buffer
  _p_d_self_step_qkv = curp;
  curp += _max_batch_size * _tw._beam_size * _tw._hidden_size * 3;
  _p_d_query_buf1 = curp;  // query buffer
  curp += _max_batch_size * _tw._beam_size * _tw._hidden_size;
  _p_d_query_buf2 = curp;  // query buffer
  curp +=
      _max_batch_size * _tw._beam_size * max(_tw._hidden_size, _tw._inner_size);
  _p_d_c = curp;
  curp += _max_batch_size * _tw._head_num * _tw._beam_size * _tw._max_step;

  // for beam search buffer
  curp = reuse_p;
  _p_d_logit_buf = curp;
  curp += _max_batch_size * _tw._beam_size * _tw._trg_vocab_size;
  _p_d_can_score = curp;
  curp += _max_batch_size * _tw._beam_size * _tw._trg_vocab_size;
  _p_d_alive_seq_probs = curp;
  curp += _max_batch_size * _tw._beam_size;
  _p_d_alive_seq_score = curp;
  curp += _max_batch_size * _tw._beam_size;

  int* pint = reinterpret_cast<int*>(curp);
  _p_d_alive_seq = pint;
  pint += _max_batch_size * _tw._beam_size * _tw._max_step;
  thrust::fill(thrust::device, _p_d_alive_seq, pint, _tw._start_id);
  _p_d_alive_seq_buf = pint;
  pint += _max_batch_size * _tw._beam_size * _tw._max_step;
  _p_d_can_idx = pint;
  pint += _max_batch_size * _tw._beam_size * _tw._trg_vocab_size;
  _p_d_can_num = pint;
  pint += _max_batch_size * _tw._beam_size + 1;
  return;
}

std::string Decoder::check() {
  if (_max_thread_per_block < _tw._hidden_size) {
    return "violate hidden_size <= max_thread_per_block";
  }
  if (_tw._inner_size % _max_thread_per_block != 0 ||
      _tw._inner_size < _max_thread_per_block) {
    return "violate inner_size >= max_thread_per_block and inner_size % "
           "_max_thread_per_block = 0";
  }
  if (_p_d_trg_emb_wei.size() != 7) {
    return "violate p_d_trg_emb_wei.size() = 7";
  }
  if (_p_d_dec_wei.size() != _tw._weight_per_dec_layer * _tw._n_dec_layer) {
    return "violate p_d_dec_wei.size() = weight_per_dec_layer * n_dec_layer";
  }
  return "";
}

void Decoder::run_one_infer(int batch_size, int batch_seq_len) {
  _batch_size = batch_size;
  _batch_seq_len = batch_seq_len;
  _batch_token_num = batch_size * batch_seq_len;
  _step_token_num = batch_size * _tw._beam_size;
  _batch_max_decode_length =
      min(_tw._max_step, batch_seq_len + _tw._extra_decode_length) - 1;
  project_encoder_output();
  cudaMemcpy(_p_d_alive_seq_probs, _h_alive_seq_probs.data(),
             sizeof(float) * _batch_size * _tw._beam_size,
             cudaMemcpyHostToDevice);
  // max _batch_max_decode_length 254
  for (_cur_step = 0; _cur_step < _batch_max_decode_length; _cur_step++) {
    // for(_cur_step = 0; _cur_step < 10; _cur_step++) {
    // std::cout<<"run step " << _cur_step << std::endl;
    if (run_step()) {
      break;
    }
  }
  // max _cur_step 254
  if (_tw._length_penalty >= 0.f || _cur_step == _batch_max_decode_length) {
    ker_write_trg_tokenid_pos_penalty<<<_batch_size, _cur_step + 1>>>(
        _p_d_alive_seq, _p_d_result, _tw._max_step, _tw._beam_size);
  } else {
    ker_write_trg_tokenid_neg_penalty<<<_batch_size, _cur_step + 1>>>(
        _p_d_alive_seq, _p_d_alive_seq_score, _p_d_result, _tw._max_step, _tw._beam_size,
	_tw._trg_vocab_size);
  }
#ifdef DEBUG_RESULT
  for(int i=0; i<_batch_size; i++) {
    print_vec(_p_d_result + i * (_cur_step + 1), "finial res", _cur_step + 1);
  }
#endif
  return;
}

void Decoder::project_encoder_output() {
#ifdef DEBUG_TIME
  auto start = std::chrono::high_resolution_clock::now();
#endif
  int kv_dim = _tw._hidden_size * 2 * _tw._n_dec_layer;
  CUBLAS_CALL(cublasSgemm(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, kv_dim, _batch_token_num, _tw._hidden_size,
      &_fone, _p_d_trg_emb_wei[4], kv_dim, _p_d_encoder_output,
      _tw._hidden_size, &_fzero, _p_d_encoder_out_buf, kv_dim));
  // _p_d_encoder_out_buf: [batch_size, batch_seq_len, layer_num, 2,
  // hidden_size]
  ker_arrange_encdec_kv<<<dim3(_batch_token_num, _tw._n_dec_layer * 2),
                          _tw._hidden_size>>>(
      _p_d_encoder_out_buf, _p_d_trg_emb_wei[5], _p_d_encdec_k_bgeem[0],
      _p_d_encdec_v_bgeem[0], _layer_size_encdec_k, _batch_seq_len,
      _tw._dim_per_head, _tw._head_num);
#ifdef DEBUG_TIME
  print_time_duration(start, "encode output project");
#endif
  return;
}

bool Decoder::run_step() {
  embedding();
  decoder_stack();
  return beam_search();
}

void Decoder::embedding() {
// _p_d_trg_emb_wei: {token_emb, position_emb, norm_scale, norm_bias,
// enc_out_kernel_kv, enc_out_bias_kv, logit_bias}
#ifdef DEBUG_TIME
  auto start = std::chrono::high_resolution_clock::now();
#endif
  ker_dec_embedding<<<_step_token_num, _tw._hidden_size>>>(
      _p_d_trg_emb_wei[0], _p_d_trg_emb_wei[1], _p_d_alive_seq,
      _p_d_cur_step_query, _cur_step, _tw._max_step, _tw._trg_vocab_size);
#ifdef DEBUG_TIME
  print_time_duration(start, "trg embedding");
#endif
  return;
}

void Decoder::decoder_stack() {
// _p_d_dec_wei = {self_norm_scale, self_norm_bias,
// self_qkv_kernel, self_qkv_bias, self_output_kernel, self_output_bias
// encdec_norm_scale, encdec_norm_bias,
// encdec_q_kernel, encdec_q_bias, encdec_output_kernel, encdec_output_bias
// ffn_norm_scale, ffn_norm_bias, ffn_first_kernel, ffn_first_bias,
// ffn_second_kernel, ffn_second_bias} * encoder_layer_num
#ifdef DEBUG_TIME
  auto start = std::chrono::high_resolution_clock::now();
#endif
  for (_layer_id = 0; _layer_id < _tw._n_dec_layer; _layer_id++) {
    // std::cout<<"run layer " << _layer_id << std::endl;

    _weight_offset = _layer_id * _tw._weight_per_dec_layer;

    self_attention();

    encdec_attention();

    ffn_add_norm();
  }
  ker_norm_layer<<<_step_token_num, _tw._hidden_size>>>(
      _p_d_cur_step_query, _p_d_trg_emb_wei[2], _p_d_trg_emb_wei[3]);
#ifdef DEBUG_TIME
  print_time_duration(start, "decoder all");
#endif
  return;
}

void Decoder::self_attention() {
  // step 0. layer_norm
  ker_norm_layer3<<<_step_token_num, _tw._hidden_size>>>(
      _p_d_cur_step_query, _p_d_query_buf1, _p_d_dec_wei[_weight_offset],
      _p_d_dec_wei[_weight_offset + 1], _p_d_dec_wei[_weight_offset + 5]);
#ifdef DEBUG_RESULT
  print_vec(_p_d_cur_step_query, "self_attention:ln output batch0, beam0", 10);
#endif

  // step 1. qkv = q * qkv_wei
  CUBLAS_CALL(cublasSgemm(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size * 3, _step_token_num,
      _tw._hidden_size, &_fone, _p_d_dec_wei[_weight_offset + 2],
      _tw._hidden_size * 3, _p_d_query_buf1, _tw._hidden_size, &_fzero,
      _p_d_self_step_qkv, _tw._hidden_size * 3));
  ker_arrange_decself_qkv<<<dim3(_step_token_num, 3), _tw._hidden_size>>>(
      _p_d_self_step_qkv, _p_d_dec_wei[_weight_offset + 3], _p_d_query_buf1,
      _p_d_self_k_bgeem1[_layer_id], _p_d_self_v_bgeem1[_layer_id],
      _tw._head_num, _tw._dim_per_head, _tw._max_step, _cur_step);
  // step 2. correlation = q * k
  CUBLAS_CALL(cublasSgemmStridedBatched(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _cur_step + 1, 1, _tw._dim_per_head,
      &_atten_scaler, _p_d_self_k_bgeem1[_layer_id], _tw._max_step,
      _tw._max_step * _tw._dim_per_head, _p_d_query_buf1, _tw._dim_per_head,
      _tw._dim_per_head, &_fzero, _p_d_c, _cur_step + 1, _cur_step + 1,
      _step_token_num * _tw._head_num));

#ifdef DEBUG_RESULT
  print_vec(_p_d_c + 4*(_cur_step + 1), "self_attention:correlation:before batch0, beam0",
		  (_cur_step + 1));
#endif
  ker_correlation_softmax_decself<<<_step_token_num * _tw._head_num,
                                    _cur_step + 1>>>(_p_d_c);
#ifdef DEBUG_RESULT
  print_vec(_p_d_c + 4*(_cur_step + 1), "self_attention:correlation:after batch0, beam0",
		  (_cur_step + 1));
#endif
  // step 3. q = correlation * v
  CUBLAS_CALL(cublasSgemmStridedBatched(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._dim_per_head, 1, _cur_step + 1,
      &_fone, _p_d_self_v_bgeem1[_layer_id], _tw._dim_per_head,
      _tw._max_step * _tw._dim_per_head, _p_d_c, _cur_step + 1, _cur_step + 1,
      &_fzero, _p_d_query_buf1, _tw._dim_per_head, _tw._dim_per_head,
      _step_token_num * _tw._head_num));
  // step 4. q = ori_q + q * output_wei
  CUBLAS_CALL(cublasSgemm(_hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size,
                          _step_token_num, _tw._hidden_size, &_fone,
                          _p_d_dec_wei[_weight_offset + 4], _tw._hidden_size,
                          _p_d_query_buf1, _tw._hidden_size, &_fone,
                          _p_d_cur_step_query, _tw._hidden_size));
#ifdef DEBUG_RESULT
  CUBLAS_CALL(cublasSgemm(_hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size,
                          _step_token_num, _tw._hidden_size, &_fone,
                          _p_d_dec_wei[_weight_offset + 4], _tw._hidden_size,
                          _p_d_query_buf1, _tw._hidden_size, &_fzero,
                          _p_d_logit_buf, _tw._hidden_size));
  print_vec(_p_d_dec_wei[_weight_offset + 4], "self_attention:output_project_wei output batch0, beam0", 10);
  print_vec(_p_d_query_buf1 + 256, "self_attention:c*v output batch0, beam0", 66);
  print_vec(_p_d_logit_buf, "self_attention:output_project output batch0, beam0", 10);
  print_vec(_p_d_cur_step_query, "self_attention output batch0, beam0", 10);
#endif
}

void Decoder::encdec_attention() {
  // step 0. layer_norm
  ker_norm_layer3<<<_step_token_num, _tw._hidden_size>>>(
      _p_d_cur_step_query, _p_d_query_buf1, _p_d_dec_wei[_weight_offset + 6],
      _p_d_dec_wei[_weight_offset + 7], _p_d_dec_wei[_weight_offset + 11]);
  // step 1. q = q * q_wei
  CUBLAS_CALL(cublasSgemm(_hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size,
                          _step_token_num, _tw._hidden_size, &_fone,
                          _p_d_dec_wei[_weight_offset + 8], _tw._hidden_size,
                          _p_d_query_buf1, _tw._hidden_size, &_fzero,
                          _p_d_query_buf2, _tw._hidden_size));
  ker_arrange_encdec_q<<<_step_token_num, _tw._hidden_size>>>(
      _p_d_query_buf2, _p_d_dec_wei[_weight_offset + 9], _p_d_query_buf1,
      _tw._beam_size, _tw._dim_per_head, _tw._head_num);
  // step 2. correlation = q * k
  CUBLAS_CALL(cublasSgemmStridedBatched(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _batch_seq_len, _tw._beam_size,
      _tw._dim_per_head, &_atten_scaler, _p_d_encdec_k_bgeem[_layer_id],
      _batch_seq_len, _batch_seq_len * _tw._dim_per_head, _p_d_query_buf1,
      _tw._dim_per_head, _tw._beam_size * _tw._dim_per_head, &_fzero, _p_d_c,
      _batch_seq_len, _tw._beam_size * _batch_seq_len,
      _batch_size * _tw._head_num));

  ker_correlation_softmax_encdec<<<
      dim3(_batch_size, _tw._head_num * _tw._beam_size), _batch_seq_len>>>(
      _p_d_c, _p_d_padding_mask);

  // step 3. q = correlation * v
  CUBLAS_CALL(cublasSgemmStridedBatched(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._dim_per_head, _tw._beam_size,
      _batch_seq_len, &_fone, _p_d_encdec_v_bgeem[_layer_id], _tw._dim_per_head,
      _batch_seq_len * _tw._dim_per_head, _p_d_c, _batch_seq_len,
      _tw._beam_size * _batch_seq_len, &_fzero, _p_d_query_buf1,
      _tw._dim_per_head, _tw._beam_size * _tw._dim_per_head,
      _batch_size * _tw._head_num));
  ker_arrange_atten_output<<<_step_token_num, _tw._hidden_size>>>(
      _p_d_query_buf1, _p_d_query_buf2, _tw._beam_size, _tw._dim_per_head,
      _tw._head_num);
  // step 4. q = ori_q + q * output_wei
  CUBLAS_CALL(cublasSgemm(_hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size,
                          _step_token_num, _tw._hidden_size, &_fone,
                          _p_d_dec_wei[_weight_offset + 10], _tw._hidden_size,
                          _p_d_query_buf2, _tw._hidden_size, &_fone,
                          _p_d_cur_step_query, _tw._hidden_size));
#ifdef DEBUG_RESULT
  print_vec(_p_d_cur_step_query, "encdec output batch0, beam0", 10);
#endif
  return;
}

void Decoder::ffn_add_norm() {
  // step 0. layer_norm
  ker_norm_layer3<<<_step_token_num, _tw._hidden_size>>>(
      _p_d_cur_step_query, _p_d_query_buf1, _p_d_dec_wei[_weight_offset + 12],
      _p_d_dec_wei[_weight_offset + 13], _p_d_dec_wei[_weight_offset + 17]);
  CUBLAS_CALL(cublasSgemm(_hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._inner_size,
                          _step_token_num, _tw._hidden_size, &_fone,
                          _p_d_dec_wei[_weight_offset + 14], _tw._inner_size,
                          _p_d_query_buf1, _tw._hidden_size, &_fzero,
                          _p_d_query_buf2, _tw._inner_size));
  kerBiasRelu<<<dim3(_step_token_num, _tw._inner_size / _max_thread_per_block),
                _max_thread_per_block>>>(
      _p_d_query_buf2, _p_d_dec_wei[_weight_offset + 15], _tw._inner_size);
  CUBLAS_CALL(cublasSgemm(_hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size,
                          _step_token_num, _tw._inner_size, &_fone,
                          _p_d_dec_wei[_weight_offset + 16], _tw._hidden_size,
                          _p_d_query_buf2, _tw._inner_size, &_fone,
                          _p_d_cur_step_query, _tw._hidden_size));
#ifdef DEBUG_RESULT
  print_vec(_p_d_cur_step_query, "ffn_add output batch0, beam0", 10);
#endif
  return;
}

bool Decoder::beam_search() {
#ifdef DEBUG_TIME
  auto start = std::chrono::high_resolution_clock::now();
#endif
  /* step 0. liner project, share weight with target embedding */
  CUBLAS_CALL(cublasSgemm(_hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._trg_vocab_size,
                          _step_token_num, _tw._hidden_size, &_output_scaler,
                          _p_d_trg_emb_wei[0], _tw._trg_vocab_size,
                          _p_d_cur_step_query, _tw._hidden_size, &_fzero,
                          _p_d_logit_buf, _tw._trg_vocab_size));
#ifdef DEBUG_RESULT
  print_vec(_p_d_cur_step_query, "decoder output batch0, beam0", 10);
  //print_vec(_p_d_cur_step_query + _tw._hidden_size * 4,
  //          "decoder output batch1, beam0", 10);
  //print_vec(_p_d_cur_step_query + _tw._hidden_size * 8,
  //          "decoder output batch2, beam0", 10);
  //print_vec(_p_d_cur_step_query + _tw._hidden_size * 12,
  //          "decoder output batch3, beam0", 10);
  print_vec(_p_d_logit_buf, "logit batch0, beam0", 10);
  print_vec(_p_d_logit_buf + _tw._trg_vocab_size * 1, "logit batch0, beam1",
            10);
  print_vec(_p_d_logit_buf + _tw._trg_vocab_size * 2, "logit batch0, beam2",
            10);
  print_vec(_p_d_logit_buf + _tw._trg_vocab_size * 3, "logit batch0, beam3",
            10);
#endif
  /* step 1. softmax and select rough topk candidate for every batch item */
  cudaMemset(_p_d_can_num, 0, sizeof(int));
  update_new_seq_probs();

  /* step 2. sort the candidate with their probability */
  cudaMemcpy(&_h_can_num_batch, _p_d_can_num, sizeof(int),
             cudaMemcpyDeviceToHost);
  if (_h_can_num_batch < _cub_sort_buffer_bytes / 160) {
    CUDA_CALL(cub::DeviceRadixSort::SortPairsDescending(
        _p_d_logit_buf, _cub_sort_buffer_bytes, _p_d_can_score, _p_d_can_score,
        _p_d_can_idx, _p_d_can_idx, _h_can_num_batch));
  } else {
    thrust::sort_by_key(thrust::device, _p_d_can_score,
                        _p_d_can_score + _h_can_num_batch, _p_d_can_idx,
                        thrust::greater<float>());
  }


  /*
    step 3. refresh alive_seq, seq_probs, seq_score, num_finish_beam
    based on sorted candidate
  */
  cudaMemset(_p_d_can_num, 0, sizeof(int));
  ker_refresh_result<<<dim3(_batch_size, _tw._beam_size), _tw._max_step>>>(
      _p_d_can_idx, _p_d_can_score, _p_d_can_num + 1, _p_d_alive_seq,
      _p_d_alive_seq_buf, _p_d_alive_seq_probs, _p_d_alive_seq_score,
      _p_d_can_num, _tw._trg_vocab_size, _cur_step, _h_length_norm[_cur_step]);
  int* tmp = _p_d_alive_seq_buf;
  _p_d_alive_seq_buf = _p_d_alive_seq;
  _p_d_alive_seq = tmp;
  cudaMemcpy(&_h_can_num_batch, _p_d_can_num, sizeof(int),
             cudaMemcpyDeviceToHost);
#ifdef DEBUG_RESULT
  print_vec(_p_d_alive_seq_probs, "seq probs",
            _batch_size * _tw._beam_size);
  print_vec(_p_d_alive_seq_score, "seq score",
            _batch_size * _tw._beam_size);
  print_vec(_p_d_alive_seq, "alive seq batch0, beam0", _cur_step + 2);
  print_vec(_p_d_alive_seq + _tw._max_step * 1, "alive seq batch0, beam1",
            _cur_step + 2);
  print_vec(_p_d_alive_seq + _tw._max_step * 2, "alive seq batch0, beam2",
            _cur_step + 2);
  print_vec(_p_d_alive_seq + _tw._max_step * 3, "alive seq batch0, beam3",
            _cur_step + 2);
#endif
  if (_h_can_num_batch == _step_token_num) {
    // std::cout<<"early stop beam search!" <<std::endl;
    return true;
  }


  /* step 4. refresh cache: k, v of self attention */
  if (_cur_step > 0) {
    ker_refresh_cache<<<dim3(_tw._n_dec_layer * (_cur_step + 1),
                             _step_token_num * 2),
                        _tw._hidden_size>>>(
        _p_d_can_num + 1, _p_d_can_idx, _p_d_self_k_bgeem1[0],
        _p_d_self_v_bgeem1[0], _p_d_self_k_bgeem2[0], _p_d_self_v_bgeem2[0],
        _layer_size_self_k, _tw._beam_size, _tw._dim_per_head, _tw._head_num,
        _tw._trg_vocab_size, _cur_step, _tw._max_step);
    float** ftmp = _p_d_self_k_bgeem2;
    _p_d_self_k_bgeem2 = _p_d_self_k_bgeem1;
    _p_d_self_k_bgeem1 = ftmp;
    ftmp = _p_d_self_v_bgeem2;
    _p_d_self_v_bgeem2 = _p_d_self_v_bgeem1;
    _p_d_self_v_bgeem1 = ftmp;
  }
#ifdef DEBUG_TIME
  print_time_duration(start, "beam search");
#endif
  return false;
}

void Decoder::update_new_seq_probs() {
#ifdef DEBUG_TIME
  auto start = std::chrono::high_resolution_clock::now();
#endif
  if (_tw._beam_size == 1)
    select_beam_rough_topk<1><<<_step_token_num, _max_thread_per_block>>>(
        _p_d_logit_buf, _p_d_trg_emb_wei[6], _p_d_alive_seq_probs,
	_p_d_alive_seq_score, _p_d_alive_seq, _p_d_can_idx, _p_d_can_score, _p_d_can_num,
        _tw._trg_vocab_size, _tw._max_step, _h_length_norm[_cur_step], _cur_step);
  if (_tw._beam_size == 2)
    select_beam_rough_topk<2><<<_step_token_num, _max_thread_per_block>>>(
        _p_d_logit_buf, _p_d_trg_emb_wei[6], _p_d_alive_seq_probs,
	_p_d_alive_seq_score, _p_d_alive_seq, _p_d_can_idx, _p_d_can_score, _p_d_can_num,
        _tw._trg_vocab_size, _tw._max_step, _h_length_norm[_cur_step], _cur_step);
  if (_tw._beam_size == 4)
    select_beam_rough_topk<4><<<_step_token_num, _max_thread_per_block>>>(
        _p_d_logit_buf, _p_d_trg_emb_wei[6], _p_d_alive_seq_probs,
	_p_d_alive_seq_score, _p_d_alive_seq, _p_d_can_idx, _p_d_can_score, _p_d_can_num,
        _tw._trg_vocab_size, _tw._max_step, _h_length_norm[_cur_step], _cur_step);
  if (_tw._beam_size == 8)
    select_beam_rough_topk<8><<<_step_token_num, _max_thread_per_block>>>(
        _p_d_logit_buf, _p_d_trg_emb_wei[6], _p_d_alive_seq_probs,
	_p_d_alive_seq_score, _p_d_alive_seq, _p_d_can_idx, _p_d_can_score, _p_d_can_num,
        _tw._trg_vocab_size, _tw._max_step, _h_length_norm[_cur_step], _cur_step);
#ifdef DEBUG_RESULT
  print_vec(_p_d_can_num, "beam can num",
            _batch_size * _tw._beam_size + 1);
#endif
  thrust::exclusive_scan(thrust::device, _p_d_can_num + 1,
                         _p_d_can_num + 1 + _step_token_num, _p_d_can_num + 1);
#ifdef DEBUG_TIME
  print_time_duration(start, "update seq probs");
#endif
  return;
}

}  // namespace nmt
}  // namespace lab
