#include <cub/cub.cuh>

#include "src/custom/transformer/kernels/nmtKernels.h"
#include "src/custom/transformer/model/decoder.h"
#include "src/custom/transformer/util.h"

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
      _tw(tw),
      _p_d_trg_emb_wei(tw.get_trg_emb_wei()),
      _p_d_dec_wei(tw.get_dec_wei()),
      _hd(hd),
      _d_alive_seq_probs(max_batch_size * tw._beam_size, 0.f),
      _d_can_probs(max_batch_size * tw._beam_size * tw._trg_vocab_size, 0.f),
      _d_can_idx(max_batch_size * tw._beam_size * tw._trg_vocab_size, 0),
      _d_finished_scores(max_batch_size * (tw._beam_size + 1), 0.f),
      _d_can_num(max_batch_size * tw._beam_size + 1, 0),
      // tw._n_dec_layer * max_batch_size * tw._max_step * tw._hidden_size * 2
      // max_batch_size * beam_size * trg_vocab_size
      // max_batch_size * beam_size * tw._inner_size * 2 +
      // max_batch_size * beam_size * max_step * max_step
      _d_buf(
          tw._n_dec_layer * max_batch_size * tw._max_step * tw._hidden_size * 2,
          0.f),
      _d_alive_seq(max_batch_size * tw._beam_size * tw._max_step * 2 +
                       max_batch_size * tw._max_step,
                   tw._start_id),
      _d_cur_step_query(max_batch_size * tw._beam_size * tw._hidden_size, 0.f),
      _layer_size_encdec_k(max_batch_size * tw._max_step * tw._hidden_size),
      _layer_size_self_k(max_batch_size * tw._max_step * tw._hidden_size *
                         tw._beam_size),
      _fone(1.f),
      _fzero(0.f),
      _atten_scaler(sqrt(1.f / tw._dim_per_head)),
      _output_scaler(sqrt(1.f / tw._hidden_size)),
      _beam_length_alpha(tw._length_penalty),
      _h_alive_seq_probs(max_batch_size * tw._beam_size,
                         min_log_probability / 2),
      _h_finished_scores(max_batch_size, min_log_probability),
      _h_length_norm(tw._max_step, 0.f) {
  _p_d_alive_seq_probs = thrust::raw_pointer_cast(_d_alive_seq_probs.data());
  _p_d_can_probs = thrust::raw_pointer_cast(_d_can_probs.data());
  _p_d_can_idx = thrust::raw_pointer_cast(_d_can_idx.data());
  _p_d_finished_scores = thrust::raw_pointer_cast(_d_finished_scores.data());
  _p_d_can_num = thrust::raw_pointer_cast(_d_can_num.data());

  _p_d_buf = thrust::raw_pointer_cast(_d_buf.data());
  _p_d_alive_seq = thrust::raw_pointer_cast(_d_alive_seq.data());
  _p_d_alive_seq_buf =
      _p_d_alive_seq + max_batch_size * tw._beam_size * tw._max_step;
  _p_d_finished_seq =
      _p_d_alive_seq_buf + max_batch_size * tw._beam_size * tw._max_step;
  _p_d_cur_step_query = thrust::raw_pointer_cast(_d_cur_step_query.data());

  _d_global = thrust::device_vector<float>(
      2 * tw._n_dec_layer * (_layer_size_self_k * 2 + _layer_size_encdec_k) +
          max_batch_size * tw._beam_size * tw._hidden_size * 3,
      0.f);
  float* curp = thrust::raw_pointer_cast(_d_global.data());
  for (int i = 0; i < tw._n_dec_layer * 2; i++) {
    _p_d_self_k_bgeem.push_back(curp);
    curp += _layer_size_self_k;
  }
  _p_d_self_k_bgeem1 = _p_d_self_k_bgeem.data();
  _p_d_self_k_bgeem2 = _p_d_self_k_bgeem.data() + tw._n_dec_layer;
  for (int i = 0; i < tw._n_dec_layer * 2; i++) {
    _p_d_self_v_bgeem.push_back(curp);
    curp += _layer_size_self_k;
  }
  _p_d_self_v_bgeem1 = _p_d_self_v_bgeem.data();
  _p_d_self_v_bgeem2 = _p_d_self_v_bgeem.data() + tw._n_dec_layer;
  for (int i = 0; i < tw._n_dec_layer; i++) {
    _p_d_encdec_k_bgeem.push_back(curp);
    curp += _layer_size_encdec_k;
  }
  for (int i = 0; i < tw._n_dec_layer; i++) {
    _p_d_encdec_v_bgeem.push_back(curp);
    curp += _layer_size_encdec_k;
  }
  _p_d_self_step_qkv = curp;

  _p_d_query_buf1 = _p_d_buf;  // store result of rearrange query
  _p_d_query_buf2 = _p_d_query_buf1 + max_batch_size * tw._beam_size *
                                          max(tw._hidden_size, tw._inner_size);
  // store result of correlation * V
  _p_d_c = _p_d_query_buf2 + max_batch_size * tw._beam_size *
                                 max(tw._hidden_size, tw._inner_size);

  for (int i = 0; i < _h_alive_seq_probs.size(); i += tw._beam_size) {
    _h_alive_seq_probs[i] = 0.f;
  }

  for (int i = 0; i < _h_length_norm.size(); i++) {
    _h_length_norm[i] = length_norm(i + 1, _beam_length_alpha);
  }
  return;
}

int Decoder::compute_buffer_bytesize() {
  return 0;
}
void Decoder::init_buffer(void* pbuf) {
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
  cudaMemcpy(_p_d_finished_scores, _h_finished_scores.data(),
             sizeof(float) * _batch_size, cudaMemcpyHostToDevice);
  for (_cur_step = 0; _cur_step < _batch_max_decode_length; _cur_step++) {
  // for(_cur_step = 0; _cur_step < 20; _cur_step++) {
    // std::cout<<"run step " << _cur_step << std::endl;
    if (run_step()) {
      break;
    }
  }
  ker_write_trg_tokenid<<<_batch_size, _cur_step + 1>>>(
      _p_d_finished_seq, _p_d_result, _tw._max_step);
  return;
}

void Decoder::project_encoder_output() {
#ifdef DEBUG_TIME
  auto start = std::chrono::high_resolution_clock::now();
#endif
  int kv_dim = _tw._hidden_size * 2 * _tw._n_dec_layer;
  CUBLAS_CALL(cublasSgemm(_hd, CUBLAS_OP_N, CUBLAS_OP_N, kv_dim,
                          _batch_token_num, _tw._hidden_size, &_fone,
                          _p_d_trg_emb_wei[4], kv_dim, _p_d_encoder_output,
                          _tw._hidden_size, &_fzero, _p_d_buf, kv_dim));
  // _p_d_buf: [layer_num, batch_size, batch_seq_len, 2, hidden_size]
  ker_arrange_encdec_kv<<<dim3(_batch_token_num, _tw._n_dec_layer * 2),
                          _tw._hidden_size>>>(
      _p_d_buf, _p_d_trg_emb_wei[5], _p_d_encdec_k_bgeem[0],
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

  ker_correlation_softmax_decself<<<_step_token_num * _tw._head_num,
                                    _cur_step + 1>>>(_p_d_c);

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
  // kerBiasAdd<<<_step_token_num, _tw._hidden_size>>>(_p_d_cur_step_query,
  //       _p_d_query_buf1, _p_d_dec_wei[_weight_offset + 17]);

  return;
}

bool Decoder::beam_search() {
  /* step 0. liner project, share weight with target embedding */
  CUBLAS_CALL(cublasSgemm(_hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._trg_vocab_size,
                          _step_token_num, _tw._hidden_size, &_output_scaler,
                          _p_d_trg_emb_wei[0], _tw._trg_vocab_size,
                          _p_d_cur_step_query, _tw._hidden_size, &_fzero,
                          _p_d_buf, _tw._trg_vocab_size));

  /* step 1. softmax and select rough topk candidate for every batch item */
  cudaMemset(_p_d_can_num, 0, sizeof(int));
  update_new_seq_probs();

  /* step 2. sort the candidate with their probability */
  cudaMemcpy(&_h_can_num_batch, _p_d_can_num, sizeof(int),
             cudaMemcpyDeviceToHost);
  if (_h_can_num_batch < _cub_sort_buffer_bytes / 160) {
    CUDA_CALL(cub::DeviceRadixSort::SortPairsDescending(
        _p_d_buf, _cub_sort_buffer_bytes, _p_d_can_probs, _p_d_can_probs,
        _p_d_can_idx, _p_d_can_idx, _h_can_num_batch));
  } else {
    thrust::sort_by_key(_d_can_probs.begin(),
                        _d_can_probs.begin() + _h_can_num_batch,
                        _d_can_idx.begin(), thrust::greater<float>());
  }
  /*
    step 3. refresh finished_seq, alive_seq_probs,
    num_finish_beam, alive_seq based on sorted candidate
  */
  cudaMemset(_p_d_can_num, 0, sizeof(int));
  ker_refresh_result<<<dim3(_batch_size, _tw._beam_size), _tw._max_step>>>(
      _p_d_can_idx, _p_d_can_probs, _p_d_can_num + 1, _p_d_alive_seq,
      _p_d_alive_seq_buf, _p_d_alive_seq_probs, _p_d_finished_scores,
      _p_d_finished_scores + _batch_size, _p_d_finished_seq, _p_d_can_num,
      _tw._trg_vocab_size, _cur_step, _h_length_norm[_batch_max_decode_length]);
  int* tmp = _p_d_alive_seq_buf;
  _p_d_alive_seq_buf = _p_d_alive_seq;
  _p_d_alive_seq = tmp;

  // print_vec(_p_d_finished_scores, "finish seq scores", _batch_size *
  // (_tw._beam_size + 1)); print_vec(_p_d_finished_seq, "finish_seq, batch1",
  // _cur_step + 2); print_vec(_p_d_finished_seq+_tw._max_step, "finish_seq,
  // batch2", _cur_step + 2); print_vec(_p_d_finished_seq+_tw._max_step*2,
  // "finish_seq, batch3", _cur_step + 2);

  cudaMemcpy(&_h_can_num_batch, _p_d_can_num, sizeof(int),
             cudaMemcpyDeviceToHost);
  if (_h_can_num_batch == _step_token_num) {
    // all alive seq will not get a higher score than current best finish seq
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

  return false;
}

void Decoder::update_new_seq_probs() {
  if (_tw._beam_size == 1)
    ker_update_new_seq_probs<1><<<_step_token_num, _max_thread_per_block>>>(
        _p_d_buf, _p_d_trg_emb_wei[6], _p_d_alive_seq_probs,
        _tw._trg_vocab_size, _p_d_can_idx, _p_d_can_probs, _p_d_can_num,
        _p_d_finished_scores, _p_d_finished_scores + _batch_size,
        _h_length_norm[_cur_step]);
  if (_tw._beam_size == 2)
    ker_update_new_seq_probs<2><<<_step_token_num, _max_thread_per_block>>>(
        _p_d_buf, _p_d_trg_emb_wei[6], _p_d_alive_seq_probs,
        _tw._trg_vocab_size, _p_d_can_idx, _p_d_can_probs, _p_d_can_num,
        _p_d_finished_scores, _p_d_finished_scores + _batch_size,
        _h_length_norm[_cur_step]);
  if (_tw._beam_size == 4)
    ker_update_new_seq_probs<4><<<_step_token_num, _max_thread_per_block>>>(
        _p_d_buf, _p_d_trg_emb_wei[6], _p_d_alive_seq_probs,
        _tw._trg_vocab_size, _p_d_can_idx, _p_d_can_probs, _p_d_can_num,
        _p_d_finished_scores, _p_d_finished_scores + _batch_size,
        _h_length_norm[_cur_step]);
  if (_tw._beam_size == 8)
    ker_update_new_seq_probs<8><<<_step_token_num, _max_thread_per_block>>>(
        _p_d_buf, _p_d_trg_emb_wei[6], _p_d_alive_seq_probs,
        _tw._trg_vocab_size, _p_d_can_idx, _p_d_can_probs, _p_d_can_num,
        _p_d_finished_scores, _p_d_finished_scores + _batch_size,
        _h_length_norm[_cur_step]);
  thrust::exclusive_scan(thrust::device, _p_d_can_num + 1,
                         _p_d_can_num + 1 + _step_token_num, _p_d_can_num + 1);
  return;
}

}  // namespace nmt
}  // namespace lab
