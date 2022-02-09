#include "quant_decoder.h"

#include "../kernels/transformerKernels.h"
#include "../kernels/transformerKernels_int8.h"
#include "../kernels/embKernels_int8.h"
#include "cublas_helper.h"

/**
@file
Transformer decoder, composed by gemm lib and
  custom cuda kernel function
*/

namespace lightseq {
namespace cuda {

template <OperationType OpType_>
QuantDecoder<OpType_>::QuantDecoder(int max_batch_size,
                                    const int* p_d_padding_mask,
                                    const _DataType* p_d_encoder_output,
                                    int* p_d_result,
                                    QuantTransformerWeight<OpType_>& tw,
                                    cudaStream_t stream, cublasHandle_t hd,
                                    bool output_topk, const int* p_d_lang_id)
    : _max_batch_size(max_batch_size),
      _max_thread_per_block(1024),
      _h_can_num_batch(0),
      _cub_sort_buffer_bytes(max_batch_size * tw._beam_size *
                             tw._trg_vocab_size * sizeof(_DataType)),
      _p_d_padding_mask(p_d_padding_mask),
      _p_d_encoder_output(p_d_encoder_output),
      _p_d_result(p_d_result),
      _p_d_trg_emb_wei(tw.get_trg_emb_wei()),
      _p_d_dec_wei(tw.get_dec_wei()),
      _tw(tw),
      _stream(stream),
      _hd(hd),
      _output_topk(output_topk),
      _p_d_lang_id(p_d_lang_id),  // source token id
      _layer_size_encdec_k(max_batch_size * tw._max_step * tw._hidden_size),
      _layer_size_self_k(max_batch_size * tw._max_step * tw._hidden_size *
                         tw._beam_size),
      _type_one(1.f),
      _type_zero(0.f),
      _fzero(0.f),

      _trg_scaled_emb_clip_max(tw.get_trg_scaled_emb_clip_max()),
      _output_ln_clip_max(tw.get_output_ln_clip_max()),
      _logits_clip_max(tw.get_logits_clip_max()),
      _encode_output_project_kernel_kv_clip_max(
          tw.get_encode_output_project_kernel_kv_clip_max()),
      _dec_clip_max(tw.get_dec_clip_max()),
      _ione((int32_t)1),
      _izero((int32_t)0),

      _atten_scaler(sqrt(1.f / tw._dim_per_head)),
      _logit_scaler(_tw._no_scale_embedding ? 1.f
                                            : sqrt(1.f / tw._hidden_size)),
      _h_alive_seq_probs(max_batch_size * tw._beam_size,
                         min_log_probability / 2),
      _h_length_norm(tw._max_step, 1.f),
      _h_unfinished(1) {
  for (int i = 0; i < _h_alive_seq_probs.size(); i += tw._beam_size) {
    _h_alive_seq_probs[i] = 0.f;
  }
  if (tw._length_penalty >= 0) {
    for (int i = 0; i < _h_length_norm.size(); i++) {
      _h_length_norm[i] = length_norm(i + 1, tw._length_penalty);
    }
  }
  CHECK_GPU_ERROR(cublasLtCreate(&_cublas_lt_handle));
  return;
}

/**
Compute GPU memory size needed by transformer decoder,
  to see how these memory is used, checkout init_buffer() for detail
*/
template <OperationType OpType_>
long QuantDecoder<OpType_>::compute_buffer_bytesize() {
  return 0;
}

/**
Init the GPU memory pointer which point to
  the memory buffer needed by decoder.
These buffer are used during custom cuda kernel function,
  find the corresponding function to see how these buffer are used
*/
template <OperationType OpType_>
void QuantDecoder<OpType_>::init_buffer() {
  std::cout << "decoder buffer init start" << std::endl;

  // malloc activations and cache
  int temp_size = _tw._n_dec_layer * 2 * _layer_size_encdec_k;
  _DataType *temp, *sliding_temp;
  CHECK_GPU_ERROR(cudaMalloc(&temp, temp_size * sizeof(_DataType)));
  sliding_temp = temp;
  for (int i = 0; i < _tw._n_dec_layer; i++) {
    // encoder ouput after project, the "key" of enc_dec attention
    _p_d_encdec_k_bgeem.push_back(sliding_temp);
    sliding_temp += _layer_size_encdec_k;
  }
  for (int i = 0; i < _tw._n_dec_layer; i++) {
    // encoder ouput after project, the "value" of enc_dec attention
    _p_d_encdec_v_bgeem.push_back(sliding_temp);
    sliding_temp += _layer_size_encdec_k;
  }

  CHECK_GPU_ERROR(cudaMalloc(
      &_p_d_cur_step_query,
      _max_batch_size * _tw._beam_size * _tw._hidden_size * sizeof(_DataType)));
  CHECK_GPU_ERROR(cudaMalloc(
      &_p_d_query_buf1,
      _max_batch_size * _tw._beam_size * _tw._hidden_size * sizeof(_DataType)));
  CHECK_GPU_ERROR(cudaMalloc(&_p_d_c, _max_batch_size * _tw._head_num *
                                          _tw._beam_size * _tw._max_step *
                                          sizeof(_DataType)));
  CHECK_GPU_ERROR(cudaMalloc(
      &_p_d_can_score,
      _max_batch_size * _tw._beam_size * _tw._trg_vocab_size * sizeof(float)));
  CHECK_GPU_ERROR(cudaMalloc(&_p_d_alive_seq_probs,
                             _max_batch_size * _tw._beam_size * sizeof(float)));
  CHECK_GPU_ERROR(cudaMalloc(&_p_d_alive_seq_score,
                             _max_batch_size * _tw._beam_size * sizeof(float)));
  CHECK_GPU_ERROR(cudaMalloc(&_p_d_alive_seq, _max_batch_size * _tw._beam_size *
                                                  _tw._max_step * sizeof(int)));
  CHECK_GPU_ERROR(cudaMalloc(
      &_p_d_alive_seq_buf,
      _max_batch_size * _tw._beam_size * _tw._max_step * sizeof(int)));
  CHECK_GPU_ERROR(cudaMalloc(
      &_p_d_can_idx,
      _max_batch_size * _tw._beam_size * _tw._trg_vocab_size * sizeof(int)));
  CHECK_GPU_ERROR(cudaMalloc(
      &_p_d_can_num, (_max_batch_size * _tw._beam_size + 1) * sizeof(int)));

  std::vector<int> start_id_vec(
      _max_batch_size * _tw._beam_size * _tw._max_step, _tw._start_id);
  CHECK_GPU_ERROR(cudaMemcpyAsync(_p_d_alive_seq, start_id_vec.data(),
                                  sizeof(int) * start_id_vec.size(),
                                  cudaMemcpyHostToDevice, _stream));

  CHECK_GPU_ERROR(cudaMemcpyAsync(_p_d_alive_seq_buf, start_id_vec.data(),
                                  sizeof(int) * start_id_vec.size(),
                                  cudaMemcpyHostToDevice, _stream));

  CHECK_GPU_ERROR(cudaMalloc(&_p_d_sample_unfinished, sizeof(int)));
  CHECK_GPU_ERROR(
      cudaMalloc(&_p_d_curandstate, _max_batch_size * sizeof(curandState)));
  ker_curand_setup<<<_max_batch_size, 1, 0, _stream>>>(_p_d_curandstate);

  int max_batch_dim =
      _max_batch_size * _tw._beam_size *
      round_up(std::max(_tw._inner_size, _tw._hidden_size * 3), 32);
  CHECK_GPU_ERROR(
      cudaMalloc(&_int8_ffn_in_buf, max_batch_dim * sizeof(int8_t)));
  CHECK_GPU_ERROR(cudaMalloc(
      &_int32_ffn_out_buf,
      std::max(std::max(max_batch_dim, _max_batch_size * _tw._beam_size *
                                           _tw._head_num * _tw._max_step),
               round_up(_tw._trg_vocab_size, 32) * _tw._beam_size *
                   _max_batch_size) *
          sizeof(int32_t)));
  CHECK_GPU_ERROR(
      cudaMalloc(&_int8_ffn_out_buf,
                 std::max(max_batch_dim, round_up(_tw._trg_vocab_size, 32) *
                                             _tw._beam_size * _max_batch_size) *
                     sizeof(int8_t)));

  // malloc embeddings
  CHECK_GPU_ERROR(
      cudaMalloc(&_int8_p_d_trg_emb_wei,
                 _tw._trg_vocab_size * _tw._hidden_size * sizeof(int8_t)));
  quantize_weight(_p_d_trg_emb_wei[0], _int8_p_d_trg_emb_wei, _tw._hidden_size,
                  _tw._trg_vocab_size, _quant_range / _trg_scaled_emb_clip_max,
                  _stream, _cublas_lt_handle);
  CHECK_GPU_ERROR(
      cudaMalloc(&_int8_p_d_trg_emb_bottom_wei,
                 _tw._trg_vocab_size * _tw._hidden_size * sizeof(int8_t)));
  quantize_weight(_p_d_trg_emb_wei[0], _int8_p_d_trg_emb_bottom_wei,
                  _tw._hidden_size, _tw._trg_vocab_size,
                  _quant_range / _trg_scaled_emb_clip_max, _stream,
                  _cublas_lt_handle, kRowMajor);
  _p_device_emb.push_back(nullptr);
  _p_device_emb.push_back(
      to_gpu(_p_d_trg_emb_wei[1], _tw._max_step * _tw._hidden_size, _stream));
  _p_device_emb.push_back(
      to_gpu(_p_d_trg_emb_wei[2], _tw._hidden_size, _stream));
  _p_device_emb.push_back(
      to_gpu(_p_d_trg_emb_wei[3], _tw._hidden_size, _stream));
  _p_device_emb.push_back(to_gpu(
      _p_d_trg_emb_wei[4],
      _tw._hidden_size * _tw._hidden_size * 2 * _tw._n_dec_layer, _stream));
  _p_device_emb.push_back(to_gpu(
      _p_d_trg_emb_wei[5], _tw._hidden_size * 2 * _tw._n_dec_layer, _stream));
  _p_device_emb.push_back(
      to_gpu(_p_d_trg_emb_wei[6], _tw._trg_vocab_size, _stream));
  if (_tw._multilg_type != 0) {
    _p_device_emb.push_back(
        to_gpu(_p_d_trg_emb_wei[7], _tw._hidden_size, _stream));
  } else {
    _p_device_emb.push_back(nullptr);
  }

  // malloc reused kv cache and encdec output
  // _p_d_encoder_out_buf max size: _tw._hidden_size * 2 * _tw._n_dec_layer *
  // _max_batch_size * _max_step * sizeof(T)
  // so when fp16 their max size is same.
  int8_t* self_kv_cache_buffer;
  int8_t* sliding_p;
  CHECK_GPU_ERROR(
      cudaMalloc(&self_kv_cache_buffer,
                 _layer_size_self_k * _tw._n_dec_layer * 4 * sizeof(int8_t)));

  int encoder_out_size = _tw._hidden_size * 2 * _tw._n_dec_layer *
                         _max_batch_size * _tw._max_step * sizeof(_DataType);
  if (encoder_out_size <=
      _layer_size_self_k * _tw._n_dec_layer * 4 * sizeof(int8_t)) {
    _p_d_encoder_out_buf = reinterpret_cast<_DataType*>(self_kv_cache_buffer);
  } else {
    CHECK_GPU_ERROR(cudaMalloc(&_p_d_encoder_out_buf, encoder_out_size));
  }

  sliding_p = self_kv_cache_buffer;
  for (int i = 0; i < _tw._n_dec_layer * 2; i++) {
    _p_d_self_k_cache.push_back(sliding_p);
    sliding_p += _layer_size_self_k;
  }
  for (int i = 0; i < _tw._n_dec_layer * 2; i++) {
    _p_d_self_v_cache.push_back(sliding_p);
    sliding_p += _layer_size_self_k;
  }
  _p_d_self_k_cache1 = _p_d_self_k_cache.data();
  _p_d_self_k_cache2 = _p_d_self_k_cache.data() + _tw._n_dec_layer;
  _p_d_self_v_cache1 = _p_d_self_v_cache.data();
  _p_d_self_v_cache2 = _p_d_self_v_cache.data() + _tw._n_dec_layer;

  // malloc weights
  _int8_p_d_dec_wei = std::vector<int8_t*>(_tw._n_dec_layer * 6);
  _scaled_ffn2_colsum = std::vector<_DataType*>(_tw._n_dec_layer);
  for (_layer_id = 0; _layer_id < _tw._n_dec_layer; _layer_id++) {
    _weight_offset = _layer_id * _tw._weight_per_dec_layer;
    // malloc quantized weights
    CHECK_GPU_ERROR(
        cudaMalloc(&_int8_p_d_dec_wei[_layer_id * 6],
                   _tw._hidden_size * 3 * _tw._hidden_size * sizeof(int8_t)));
    CHECK_GPU_ERROR(
        cudaMalloc(&_int8_p_d_dec_wei[_layer_id * 6 + 1],
                   _tw._hidden_size * _tw._hidden_size * sizeof(int8_t)));
    CHECK_GPU_ERROR(
        cudaMalloc(&_int8_p_d_dec_wei[_layer_id * 6 + 2],
                   _tw._hidden_size * _tw._hidden_size * sizeof(int8_t)));
    CHECK_GPU_ERROR(
        cudaMalloc(&_int8_p_d_dec_wei[_layer_id * 6 + 3],
                   _tw._hidden_size * _tw._hidden_size * sizeof(int8_t)));
    CHECK_GPU_ERROR(
        cudaMalloc(&_int8_p_d_dec_wei[_layer_id * 6 + 4],
                   _tw._hidden_size * _tw._inner_size * sizeof(int8_t)));
    CHECK_GPU_ERROR(
        cudaMalloc(&_int8_p_d_dec_wei[_layer_id * 6 + 5],
                   _tw._inner_size * _tw._hidden_size * sizeof(int8_t)));

    // malloc unquantized weights
    _p_device_wei.push_back(
        to_gpu(_p_d_dec_wei[_weight_offset], _tw._hidden_size, _stream));
    _p_device_wei.push_back(
        to_gpu(_p_d_dec_wei[_weight_offset + 1], _tw._hidden_size, _stream));
    _p_device_wei.push_back(nullptr);
    _p_device_wei.push_back(to_gpu(_p_d_dec_wei[_weight_offset + 3],
                                   _tw._hidden_size * 3, _stream));
    _p_device_wei.push_back(nullptr);
    _p_device_wei.push_back(
        to_gpu(_p_d_dec_wei[_weight_offset + 5], _tw._hidden_size, _stream));
    _p_device_wei.push_back(
        to_gpu(_p_d_dec_wei[_weight_offset + 6], _tw._hidden_size, _stream));
    _p_device_wei.push_back(
        to_gpu(_p_d_dec_wei[_weight_offset + 7], _tw._hidden_size, _stream));
    _p_device_wei.push_back(nullptr);
    _p_device_wei.push_back(
        to_gpu(_p_d_dec_wei[_weight_offset + 9], _tw._hidden_size, _stream));
    _p_device_wei.push_back(nullptr);
    _p_device_wei.push_back(
        to_gpu(_p_d_dec_wei[_weight_offset + 11], _tw._hidden_size, _stream));
    _p_device_wei.push_back(
        to_gpu(_p_d_dec_wei[_weight_offset + 12], _tw._hidden_size, _stream));
    _p_device_wei.push_back(
        to_gpu(_p_d_dec_wei[_weight_offset + 13], _tw._hidden_size, _stream));
    _p_device_wei.push_back(nullptr);
    _p_device_wei.push_back(
        to_gpu(_p_d_dec_wei[_weight_offset + 15], _tw._inner_size, _stream));
    _p_device_wei.push_back(nullptr);
    _p_device_wei.push_back(
        to_gpu(_p_d_dec_wei[_weight_offset + 17], _tw._hidden_size, _stream));

    quantize_weight(_p_d_dec_wei[_weight_offset + 2],
                    _int8_p_d_dec_wei[_layer_id * 6], _tw._hidden_size,
                    _tw._hidden_size * 3,
                    _quant_range / _dec_clip_max[_layer_id * 19], _stream,
                    _cublas_lt_handle);

    quantize_weight(_p_d_dec_wei[_weight_offset + 4],
                    _int8_p_d_dec_wei[_layer_id * 6 + 1], _tw._hidden_size,
                    _tw._hidden_size,
                    _quant_range / _dec_clip_max[_layer_id * 19 + 1], _stream,
                    _cublas_lt_handle, kColMajor);

    quantize_weight(_p_d_dec_wei[_weight_offset + 8],
                    _int8_p_d_dec_wei[_layer_id * 6 + 2], _tw._hidden_size,
                    _tw._hidden_size,
                    _quant_range / _dec_clip_max[_layer_id * 19 + 2], _stream,
                    _cublas_lt_handle);

    quantize_weight(_p_d_dec_wei[_weight_offset + 10],
                    _int8_p_d_dec_wei[_layer_id * 6 + 3], _tw._hidden_size,
                    _tw._hidden_size,
                    _quant_range / _dec_clip_max[_layer_id * 19 + 3], _stream,
                    _cublas_lt_handle, kColMajor);

    quantize_weight(_p_d_dec_wei[_weight_offset + 14],
                    _int8_p_d_dec_wei[_layer_id * 6 + 4], _tw._hidden_size,
                    _tw._inner_size,
                    _quant_range / _dec_clip_max[_layer_id * 19 + 4], _stream,
                    _cublas_lt_handle);

    quantize_weight(_p_d_dec_wei[_weight_offset + 16],
                    _int8_p_d_dec_wei[_layer_id * 6 + 5], _tw._inner_size,
                    _tw._hidden_size,
                    _quant_range / _dec_clip_max[_layer_id * 19 + 5], _stream,
                    _cublas_lt_handle, kColMajor);

    if (_tw._use_gelu) {
      _scaled_ffn2_colsum[_layer_id] = nullptr;
    } else {
      CHECK_GPU_ERROR(cudaMalloc(&_scaled_ffn2_colsum[_layer_id],
                                 _tw._hidden_size * sizeof(_DataType)));
      float relu_scale = _dec_clip_max[_layer_id * 19 + 11] / 2;

      _DataType* temp;
      int weight_size = _tw._inner_size * _tw._hidden_size;

      CHECK_GPU_ERROR(cudaMalloc(&temp, weight_size * sizeof(_DataType)));
      CHECK_GPU_ERROR(cudaMemcpyAsync(temp, _p_d_dec_wei[_weight_offset + 16],
                                      weight_size * sizeof(_DataType),
                                      cudaMemcpyHostToDevice, _stream));
      launch_scaled_colsum(temp, _scaled_ffn2_colsum[_layer_id],
                           _tw._inner_size, _tw._hidden_size, relu_scale,
                           _stream);
      CHECK_GPU_ERROR(cudaGetLastError());
      CHECK_GPU_ERROR(cudaFree(temp));
    }
  }

  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  CHECK_GPU_ERROR(cudaGetLastError());
  std::cout << "decoder buffer init succeed" << std::endl;
  return;
}

/**
Some requirements needed by custom cuda kernel function
*/
template <OperationType OpType_>
std::string QuantDecoder<OpType_>::check() {
  // if (_max_thread_per_block < _tw._hidden_size) {
  //   return "violate hidden_size <= max_thread_per_block";
  // }
  if (_tw._inner_size & 1) {
    return "violate inner_size % 2 = 0";
  }
  if (_tw._dim_per_head & 1) {
    return "violate dim_per_head % 2 = 0";
  }
  if (_tw._multilg_type == 0 && _p_d_trg_emb_wei.size() != 7) {
    return "violate p_d_trg_emb_wei.size() = 7";
  }
  if (_tw._multilg_type != 0 && _p_d_trg_emb_wei.size() != 8) {
    return "violate p_d_trg_emb_wei.size() = 8";
  }
  if (_p_d_dec_wei.size() != _tw._weight_per_dec_layer * _tw._n_dec_layer) {
    return "violate p_d_dec_wei.size() = weight_per_dec_layer * n_dec_layer";
  }
  bool btmp = false;
  for (int i = 1; i < 64; i *= 2) {
    if (i == _tw._beam_size) {
      btmp = true;
      break;
    }
  }
  if (!btmp) {
    return "wrong beam_size, should be 1, 2, 4, 8, 16 or 32";
  }

  std::string sampling_method = _tw._sampling_method;
  if (kSamplingMethods.find(sampling_method) == kSamplingMethods.end()) {
    return std::string("unsupported sampling_method: ") + sampling_method;
  }
  if (sampling_method == "topk" || sampling_method == "topp") {
    _output_topk = false;
  }
  if (sampling_method == "topk_greedy") {
    _output_topk = true;
  }
  if (_tw._multilg_type != 0 && _p_d_lang_id == nullptr) {
    return "lang id should not be null when multilg";
  }
  if (_tw._max_step > 1024) {
    return "max_step should not greater than 1024";
  }
  return "";
}

/**
QuantDecoder inference
*/
template <OperationType OpType_>
void QuantDecoder<OpType_>::run_one_infer(int batch_size, int batch_seq_len) {
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
  _step_token_num = batch_size * _tw._beam_size;
  _batch_max_decode_length =
      min(_tw._max_step, batch_seq_len + _tw._extra_decode_length) - 1;
  _is_sampling =
      (_tw._sampling_method == "topk" || _tw._sampling_method == "topp" ||
       _tw._sampling_method == "topk_greedy");
  if (_is_sampling) {
    _batch_max_decode_length = _tw._max_step;
  }

  project_encoder_output();  // project encoder output
  // init the first step's token id with target start_id
  CHECK_GPU_ERROR(cudaMemcpyAsync(_p_d_alive_seq_probs,
                                  _h_alive_seq_probs.data(),
                                  sizeof(float) * _batch_size * _tw._beam_size,
                                  cudaMemcpyHostToDevice, _stream));

  /* ---step2. autoregressive decoding--- */
  for (_cur_step = 0; _cur_step < _batch_max_decode_length - 1; _cur_step++) {
#ifdef DEBUG_RESULT
    std::cout << "*** run step " << _cur_step << " ***" << std::endl;
#endif
    if (run_step()) {  // one step
      break;
    }
  }

  /* ---step3. output the decoding result--- */
  if (_output_topk || _is_sampling) {
    if (_cur_step == _batch_max_decode_length) {
      _cur_step -= 1;
    }
    ker_write_topk_result<<<_batch_size * _tw._beam_size, _cur_step + 1, 0,
                            _stream>>>(
        _p_d_alive_seq, _p_d_alive_seq_score, _p_d_result, _tw._trg_vocab_size,
        _tw._max_step, _tw._beam_size, _tw._end_id);
    return;
  }
  if (_tw._length_penalty >= 0.f || _cur_step == _batch_max_decode_length) {
    ker_write_trg_tokenid_pos_penalty<<<_batch_size, _cur_step + 1, 0,
                                        _stream>>>(
        _p_d_alive_seq, _p_d_alive_seq_score, _p_d_result, _tw._max_step,
        _tw._beam_size);
  } else {
    ker_write_trg_tokenid_neg_penalty<<<_batch_size, _cur_step + 1, 0,
                                        _stream>>>(
        _p_d_alive_seq, _p_d_alive_seq_score, _p_d_result, _tw._max_step,
        _tw._beam_size, _tw._trg_vocab_size, _tw._end_id);
  }
#ifdef DEBUG_RESULT
  for (int i = 0; i < _batch_size; i++) {
    print_vec(_p_d_result + i * (_cur_step + 1), "finial res", _cur_step + 1);
  }
#endif
  return;
}

/**
Project encoder output
*/
template <OperationType OpType_>
void QuantDecoder<OpType_>::project_encoder_output() {
  int kv_dim = _tw._hidden_size * 2 * _tw._n_dec_layer;
#ifdef DEBUG_RESULT
  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  print_vec(_p_d_encoder_output, "_p_d_encoder_output(head):", 5);
  print_vec(_p_d_encoder_output + _batch_token_num * _tw._hidden_size - 5,
            "_p_d_encoder_output(tail)", 5);
  print_vec(_p_device_emb[4], "encoder project(head):", 10);
#endif
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, kv_dim, _batch_token_num, _tw._hidden_size,
      &_type_one, _p_device_emb[4], _AType, kv_dim, _p_d_encoder_output, _BType,
      _tw._hidden_size, &_type_zero, _p_d_encoder_out_buf, _CType, kv_dim,
      _computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  // _p_d_encoder_out_buf: [batch_size, batch_seq_len, layer_num, 2,
  // hidden_size]

#ifdef DEBUG_RESULT
  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  print_vec(_p_d_encoder_out_buf, "encoder out(head):", 5);
  print_vec(_p_d_encoder_out_buf +
                _batch_token_num * _tw._hidden_size * _tw._n_dec_layer - 5,
            "encoder out(tail):", 5);
#endif
  ker_arrange_encdec_kv_launcher<_DataType>(
      _batch_token_num, _tw._n_dec_layer, _tw._hidden_size, _stream,
      _p_d_encoder_out_buf, _p_device_emb[5], _p_d_encdec_k_bgeem[0],
      _p_d_encdec_v_bgeem[0], _layer_size_encdec_k, _batch_seq_len,
      _tw._dim_per_head, _tw._head_num, _max_thread_per_block);
  return;
}

/**
Decode one step
*/
template <OperationType OpType_>
bool QuantDecoder<OpType_>::run_step() {
  embedding();
  decoder_stack();
  /* --- Project hidden states to vocab logits--- */

  // _step_token_num (beam_size * batch_size) must be 4x

  cublasLtMM_withAlgo_i8IO(_int8_ffn_out_buf, 1, _step_token_num,
                           _tw._trg_vocab_size, _tw._hidden_size, 0, 0, 0,
                           _output_ln_clip_max * _trg_scaled_emb_clip_max *
                               _logit_scaler /
                               (_logits_clip_max * _quant_range),
                           _int8_ffn_in_buf, _int8_p_d_trg_emb_wei,
                           _cublas_lt_handle, _stream, false);

#ifdef DEBUG_RESULT
  for (int i = 0; i < _batch_size; i++) {       // batch_id
    for (int j = 0; j < _tw._beam_size; j++) {  // beam_id
      std::cout << "decoder output: batch-" << i << ", beam-" << j << std::endl;
      print_vec(_int8_ffn_in_buf + i * _tw._beam_size * _tw._hidden_size +
                    j * _tw._hidden_size,
                "hidden", 10);
      print_vec(_int8_ffn_out_buf + i * _tw._beam_size * _tw._trg_vocab_size +
                    j * _tw._trg_vocab_size,
                "logits", 10);
    }
  }
#endif

  if (_tw._sampling_method == "topk") {
    return sample();
  } else if (_tw._sampling_method == "topp") {
    return sample();
  } else if (_tw._sampling_method == "topk_greedy") {
    return topk_greedy_search();
  } else if (_tw._sampling_method == "beam_search") {
    return beam_search();
  } else {
    throw std::runtime_error("not supported sampling_method");
  }
}  // namespace cuda

/**
Decode embedding
*/
template <OperationType OpType_>
void QuantDecoder<OpType_>::embedding() {
  // _p_d_trg_emb_wei: {token_emb, position_emb, norm_scale, norm_bias,
  // enc_out_kernel_kv, enc_out_bias_kv, logit_bias}
  launch_dec_emb_i8I<_DataType>(
      _int8_p_d_trg_emb_bottom_wei, _p_device_emb[1], _p_d_alive_seq,
      _p_device_emb[7], _p_d_lang_id, _p_d_cur_step_query, _batch_size,
      _tw._beam_size, _tw._hidden_size, _tw._trg_vocab_size, _cur_step,
      _tw._max_step, _tw._multilg_type, _stream,
      _trg_scaled_emb_clip_max / _quant_range);
#ifdef DEBUG_RESULT
  for (int i = 0; i < _batch_size; i++) {       // batch_id
    for (int j = 0; j < _tw._beam_size; j++) {  // beam_id
      std::cout << "decoder emb: batch-" << i << ", beam-" << j << std::endl;
      print_vec(_p_d_cur_step_query + i * _tw._beam_size * _tw._hidden_size +
                    j * _tw._hidden_size,
                "emb", 10);
    }
  }
#endif
  return;
}

/**
QuantDecoder feedforward, composed by self_atten,
  enc-dec-atten, ffn
*/
template <OperationType OpType_>
void QuantDecoder<OpType_>::decoder_stack() {
  // _p_d_dec_wei = {self_norm_scale, self_norm_bias,
  // self_qkv_kernel, self_qkv_bias, self_output_kernel, self_output_bias
  // encdec_norm_scale, encdec_norm_bias,
  // encdec_q_kernel, encdec_q_bias, encdec_output_kernel, encdec_output_bias
  // ffn_norm_scale, ffn_norm_bias, ffn_first_kernel, ffn_first_bias,
  // ffn_second_kernel, ffn_second_bias} * encoder_layer_num
  for (_layer_id = 0; _layer_id < _tw._n_dec_layer; _layer_id++) {
    _weight_offset = _layer_id * _tw._weight_per_dec_layer;

    self_attention();

    encdec_attention();

    ffn_add_norm();
  }
}

/**
QuantDecoder self attention
*/
template <OperationType OpType_>
void QuantDecoder<OpType_>::self_attention() {
  if (_layer_id == 0) {
    ker_norm_layer_resual_i8O_launcher<_DataType>(
        _step_token_num, _tw._hidden_size, _stream, _p_d_cur_step_query,
        _int8_ffn_in_buf, _p_device_wei[_weight_offset],
        _p_device_wei[_weight_offset + 1], _p_device_wei[_weight_offset + 5],
        _max_thread_per_block, _quant_range / _dec_clip_max[_layer_id * 19 + 6],
        _tw._is_post_ln, true);
  }

#ifdef DEBUG_RESULT
  print_vec(_int8_ffn_in_buf, "self attn ln(head): ", 5);
  print_vec(_int8_ffn_in_buf + _step_token_num * _tw._hidden_size - 5,
            "self attn ln(tail): ", 5);
  CHECK_GPU_ERROR(cudaGetLastError());
#endif

  cublasLtMM_withAlgo_i8IO(
      _int8_ffn_out_buf, 1, _step_token_num, _tw._hidden_size * 3,
      _tw._hidden_size, 0, 0, 0,
      _dec_clip_max[_layer_id * 19] * _dec_clip_max[_layer_id * 19 + 6] /
          (_dec_clip_max[_layer_id * 19 + 12] * _quant_range),
      _int8_ffn_in_buf, _int8_p_d_dec_wei[_layer_id * 6], _cublas_lt_handle,
      _stream, false);

#ifdef DEBUG_RESULT
  print_vec(_int8_ffn_out_buf, "self qkv(head): ", 5);
  print_vec(_int8_ffn_out_buf + _step_token_num * _tw._hidden_size * 3 - 5,
            "self qkv(tail): ", 5);
#endif

  // get q, k, v by split and reshape qkv

  ker_arrange_decself_qkv_i8I_launcher<_DataType>(
      _step_token_num, _tw._hidden_size, _stream, _int8_ffn_out_buf,
      _p_device_wei[_weight_offset + 3], _int8_ffn_in_buf,
      _p_d_self_k_cache1[_layer_id], _p_d_self_v_cache1[_layer_id],
      _tw._head_num, _tw._dim_per_head, _tw._max_step, _cur_step,
      _max_thread_per_block, _dec_clip_max[_layer_id * 19 + 12] / _quant_range,
      _quant_range / _dec_clip_max[_layer_id * 19 + 18], true);

#ifdef DEBUG_RESULT
  print_vec(_int8_ffn_in_buf, "rearanged q(head): ", 5);
  print_vec(_int8_ffn_in_buf + _step_token_num * _tw._hidden_size - 5,
            "rearanged q(tail): ", 5);
  print_vec(_p_d_self_k_cache1[_layer_id], "rearanged k(head): ", 5);
  print_vec(_p_d_self_v_cache1[_layer_id], "rearanged v(head): ", 5);
  CHECK_GPU_ERROR(cudaGetLastError());
#endif

  CHECK_GPU_ERROR(cublasGemmStridedBatchedEx(
      _hd, CUBLAS_OP_T, CUBLAS_OP_N, _cur_step + 1, 1, _tw._dim_per_head,
      &_ione, _p_d_self_k_cache1[_layer_id], CUDA_R_8I, _tw._dim_per_head,
      _tw._max_step * _tw._dim_per_head, _int8_ffn_in_buf, CUDA_R_8I,
      _tw._dim_per_head, _tw._dim_per_head, &_izero, _int32_ffn_out_buf,
      CUDA_R_32I, _cur_step + 1, _tw._max_step, _step_token_num * _tw._head_num,
      CUDA_R_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

#ifdef DEBUG_RESULT
  print_vec(_int32_ffn_out_buf, "self attn q*k logits(head): ", _cur_step + 1);
  print_vec(_int32_ffn_out_buf +
                (_step_token_num * _tw._head_num - 1) * _tw._max_step,
            "self attn q*k logits(tail): ", _cur_step + 1);
  CHECK_GPU_ERROR(cudaGetLastError());
#endif

  ker_fuse_softmax_new_value_int8_launcher(
      _int32_ffn_out_buf, _p_d_self_v_cache1[_layer_id], _int8_ffn_in_buf,
      _step_token_num * _tw._head_num, _cur_step + 1, _tw._max_step,
      _tw._head_num, _tw._dim_per_head, float(_atten_scaler),
      _dec_clip_max[_layer_id * 19 + 18] / _quant_range,
      _quant_range / _dec_clip_max[_layer_id * 19 + 7], false, _stream);

#ifdef DEBUG_RESULT
  print_vec(_int8_ffn_in_buf, "self attn ffn in(head): ", 40);
  print_vec(_int8_ffn_in_buf + _step_token_num * _tw._hidden_size - 40,
            "self attn ffn in(tail): ", 40);
  CHECK_GPU_ERROR(cudaGetLastError());
#endif

  cublaslt_gemm(
      _int8_p_d_dec_wei[_layer_id * 6 + 1], _int8_ffn_in_buf, _int8_ffn_out_buf,
      1, _tw._hidden_size, _step_token_num, _tw._hidden_size, 0, 0, 0,
      _dec_clip_max[_layer_id * 19 + 1] * _dec_clip_max[_layer_id * 19 + 7] /
          (_dec_clip_max[_layer_id * 19 + 13] * _quant_range),
      _cublas_lt_handle, _stream);

#ifdef DEBUG_RESULT
  print_vec(_int8_ffn_out_buf, "self attn ffn out w/o bias(head): ", 40);
  print_vec(_int8_ffn_out_buf + _step_token_num * _tw._hidden_size - 40,
            "self attn ffn out w/o bias(tail): ", 40);
#endif

  ker_residual_bias_ln_i8I_i8O_launcher<_DataType>(
      _int8_ffn_out_buf, _p_device_wei[_weight_offset + 6],
      _p_device_wei[_weight_offset + 7], _p_device_wei[_weight_offset + 11],
      _int8_ffn_in_buf, _p_d_cur_step_query, _step_token_num, _tw._hidden_size,
      _dec_clip_max[_layer_id * 19 + 13] / _quant_range,
      _quant_range / _dec_clip_max[_layer_id * 19 + 8], _max_thread_per_block,
      _stream, _tw._is_post_ln, false, true);
}

/**
Encode-Decoder attention
*/
template <OperationType OpType_>
void QuantDecoder<OpType_>::encdec_attention() {
#ifdef DEBUG_RESULT
  print_vec(_int8_ffn_in_buf, "encdec attn ln(head): ", 5);
  print_vec(_int8_ffn_in_buf + _step_token_num * _tw._hidden_size - 5,
            "encdec attn ln(tail): ", 5);
  CHECK_GPU_ERROR(cudaGetLastError());
#endif

  cublasLtMM_withAlgo_i8IO(
      _int8_ffn_out_buf, 1, _step_token_num, _tw._hidden_size, _tw._hidden_size,
      0, 0, 0,
      _dec_clip_max[_layer_id * 19 + 2] * _dec_clip_max[_layer_id * 19 + 8] /
          (_dec_clip_max[_layer_id * 19 + 14] * _quant_range),
      _int8_ffn_in_buf, _int8_p_d_dec_wei[_layer_id * 6 + 2], _cublas_lt_handle,
      _stream, false);

#ifdef DEBUG_RESULT
  print_vec(_int8_ffn_out_buf, "encdec q(head): ", 5);
  print_vec(_int8_ffn_out_buf + _step_token_num * _tw._hidden_size - 5,
            "encdec q(tail): ", 5);
#endif

  ker_arrange_encdec_q_i8I_launcher<_DataType>(
      _step_token_num, _tw._hidden_size, _stream, _int8_ffn_out_buf,
      _p_device_wei[_weight_offset + 9], _p_d_query_buf1, _tw._beam_size,
      _tw._dim_per_head, _tw._head_num, _max_thread_per_block,
      _dec_clip_max[_layer_id * 19 + 14] / _quant_range, true);

#ifdef DEBUG_RESULT
  print_vec(_p_d_query_buf1, "rearanged q(head): ", 5);
  print_vec(_p_d_query_buf1 + _step_token_num * _tw._hidden_size - 5,
            "rearanged q(tail): ", 5);
  CHECK_GPU_ERROR(cudaGetLastError());
#endif

  /* ---step 2. correlation = q * k, perform softmax on correlation--- */
  CHECK_GPU_ERROR(cublasGemmStridedBatchedEx(
      _hd, CUBLAS_OP_T, CUBLAS_OP_N, _batch_seq_len, _tw._beam_size,
      _tw._dim_per_head, &_atten_scaler, _p_d_encdec_k_bgeem[_layer_id], _AType,
      _tw._dim_per_head, _batch_seq_len * _tw._dim_per_head, _p_d_query_buf1,
      _BType, _tw._dim_per_head, _tw._beam_size * _tw._dim_per_head,
      &_type_zero, _p_d_c, _CType, _batch_seq_len,
      _tw._beam_size * _batch_seq_len, _batch_size * _tw._head_num,
      _computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  ker_correlation_softmax_encdec_launcher<_DataType>(
      _batch_size, _tw._head_num * _tw._beam_size, _batch_seq_len, _stream,
      _p_d_c, _p_d_padding_mask);

  /* ---step 3. new_q = correlation * v--- */
  CHECK_GPU_ERROR(cublasGemmStridedBatchedEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._dim_per_head, _tw._beam_size,
      _batch_seq_len, &_type_one, _p_d_encdec_v_bgeem[_layer_id], _AType,
      _tw._dim_per_head, _batch_seq_len * _tw._dim_per_head, _p_d_c, _BType,
      _batch_seq_len, _tw._beam_size * _batch_seq_len, &_type_zero,
      _p_d_query_buf1, _CType, _tw._dim_per_head,
      _tw._beam_size * _tw._dim_per_head, _batch_size * _tw._head_num,
      _computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

#ifdef DEBUG_RESULT
  print_vec(_p_d_encdec_v_bgeem[_layer_id], "encdec attn value(head): ", 5);
  print_vec(_p_d_c, "encdec attn correlation(head): ", 5);
  print_vec(_p_d_query_buf1, "encdec attn new value(head): ", 5);

#endif

  ker_arrange_atten_output_i8O_launcher<_DataType>(
      _step_token_num, _tw._hidden_size, _stream, _p_d_query_buf1,
      _int8_ffn_in_buf, _tw._beam_size, _tw._dim_per_head, _tw._head_num,
      _max_thread_per_block, _quant_range / _dec_clip_max[_layer_id * 19 + 9],
      false);

#ifdef DEBUG_RESULT
  print_vec(_int8_ffn_in_buf, "encdec attn ffn in(head): ", 3);
  print_vec(_int8_ffn_in_buf + _step_token_num * _tw._hidden_size - 3,
            "encdec attn ffn in(tail): ", 3);
  CHECK_GPU_ERROR(cudaGetLastError());
#endif

  cublaslt_gemm(
      _int8_p_d_dec_wei[_layer_id * 6 + 3], _int8_ffn_in_buf, _int8_ffn_out_buf,
      1, _tw._hidden_size, _step_token_num, _tw._hidden_size, 0, 0, 0,
      _dec_clip_max[_layer_id * 19 + 3] * _dec_clip_max[_layer_id * 19 + 9] /
          (_dec_clip_max[_layer_id * 19 + 15] * _quant_range),
      _cublas_lt_handle, _stream);

#ifdef DEBUG_RESULT
  print_vec(_int8_ffn_out_buf, "encdec attn ffn out w/o bias(head): ", 3);
  print_vec(_int8_ffn_out_buf + _step_token_num * _tw._hidden_size - 3,
            "encdec attn ffn out w/o bias(tail): ", 3);
#endif

  ker_residual_bias_ln_i8I_i8O_launcher<_DataType>(
      _int8_ffn_out_buf, _p_device_wei[_weight_offset + 12],
      _p_device_wei[_weight_offset + 13], _p_device_wei[_weight_offset + 17],
      _int8_ffn_in_buf, _p_d_cur_step_query, _step_token_num, _tw._hidden_size,
      _dec_clip_max[_layer_id * 19 + 15] / _quant_range,
      _quant_range / _dec_clip_max[_layer_id * 19 + 10], _max_thread_per_block,
      _stream, _tw._is_post_ln, false, true);

#ifdef DEBUG_RESULT
  CHECK_GPU_ERROR(cudaGetLastError());
  print_vec(_p_d_cur_step_query, "encdec attn ffn out(head): ", 3);
  print_vec(_p_d_cur_step_query + _step_token_num * _tw._hidden_size - 3,
            "encdec attn ffn out(tail): ", 3);
#endif
  return;
}

template <OperationType OpType_>
void QuantDecoder<OpType_>::ffn_add_norm() {
#ifdef DEBUG_RESULT
  print_vec(_int8_ffn_in_buf, "ffn ln(head): ", 5);
  print_vec(_int8_ffn_in_buf + _step_token_num * _tw._hidden_size - 5,
            "ffn ln(tail): ", 5);
#endif

  cublasLtMM_withAlgo_i8IO(
      _int8_ffn_out_buf, 1, _step_token_num, _tw._inner_size, _tw._hidden_size,
      0, 0, 0,
      _dec_clip_max[_layer_id * 19 + 4] * _dec_clip_max[_layer_id * 19 + 10] /
          (_dec_clip_max[_layer_id * 19 + 16] * _quant_range),
      _int8_ffn_in_buf, _int8_p_d_dec_wei[_layer_id * 6 + 4], _cublas_lt_handle,
      _stream, false);

  if (_tw._use_gelu) {
    ker_bias_gelu_i8I_i8O_launcher<_DataType>(
        _step_token_num, _stream, _int8_ffn_out_buf, _int8_ffn_in_buf,
        _p_device_wei[_weight_offset + 15], _tw._inner_size,
        _dec_clip_max[_layer_id * 19 + 16] / _quant_range,
        _quant_range / _dec_clip_max[_layer_id * 19 + 11], true);
  } else {
    ker_bias_relu_i8I_i8O_launcher<_DataType>(
        _step_token_num, _stream, _int8_ffn_out_buf, _int8_ffn_in_buf,
        _p_device_wei[_weight_offset + 15], _tw._inner_size,
        _dec_clip_max[_layer_id * 19 + 16] / _quant_range,
        _quant_range / _dec_clip_max[_layer_id * 19 + 11],
        _dec_clip_max[_layer_id * 19 + 11], true, false, true);
  }

#ifdef DEBUG_RESULT
  print_vec(_int8_ffn_in_buf, "ffn act out(head): ", 5);
  print_vec(_int8_ffn_in_buf + _step_token_num * _tw._inner_size - 5,
            "ffn act out(tail): ", 5);
  CHECK_GPU_ERROR(cudaGetLastError());
#endif

  cublaslt_gemm(_int8_p_d_dec_wei[_layer_id * 6 + 5], _int8_ffn_in_buf,
                _int32_ffn_out_buf, 1, _tw._hidden_size, _step_token_num,
                _tw._inner_size, 0, 0, 0, 1, _cublas_lt_handle, _stream);

  const _DataType *scale_ptr, *bias_ptr, *res_bias_ptr;
  float clip_max;
  if (_layer_id == _tw._n_dec_layer - 1) {
    scale_ptr = _p_device_emb[2];
    bias_ptr = _p_device_emb[3];
    res_bias_ptr = nullptr;
    clip_max = _output_ln_clip_max;
  } else {
    scale_ptr = _p_device_wei[(_layer_id + 1) * _tw._weight_per_dec_layer];
    bias_ptr = _p_device_wei[(_layer_id + 1) * _tw._weight_per_dec_layer + 1];
    res_bias_ptr =
        _p_device_wei[(_layer_id + 1) * _tw._weight_per_dec_layer + 5];
    clip_max = _dec_clip_max[(_layer_id + 1) * 19 + 6];
  }

  ker_residual_bias_ln_i32I_i8O_launcher<_DataType>(
      _int32_ffn_out_buf, scale_ptr, bias_ptr, res_bias_ptr, _int8_ffn_in_buf,
      _p_d_cur_step_query, _step_token_num, _tw._hidden_size,
      _dec_clip_max[_layer_id * 19 + 5] * _dec_clip_max[_layer_id * 19 + 11] /
          (2 * _quant_range * _quant_range),
      _quant_range / clip_max, _max_thread_per_block, _stream, _tw._is_post_ln,
      false, true, _scaled_ffn2_colsum[_layer_id]);

#ifdef DEBUG_RESULT
  print_vec(_p_d_cur_step_query, "ffn ln(head): ", 5);
  print_vec(_p_d_cur_step_query + _step_token_num * _tw._hidden_size - 5,
            "ffn ln(tail): ", 5);
  CHECK_GPU_ERROR(cudaGetLastError());

#endif

  return;
}

template <OperationType OpType_>
bool QuantDecoder<OpType_>::sample() {
  throw std::runtime_error("QuantDecoder sample() not implemented");
  CHECK_GPU_ERROR(
      cudaMemsetAsync(_p_d_sample_unfinished, 0, sizeof(int), _stream));
  /* --- Sample new tokens from logits --- */
  if (_tw._sampling_method == "topk") {
    ker_topk_sample_launcher<_DataType>(
        _batch_size, (_cur_step + 1), _tw._max_step, 1, _max_thread_per_block,
        _stream, _p_d_logit_buf, _p_device_emb[6], _p_d_alive_seq,
        _p_d_alive_seq_buf, _tw._trg_vocab_size, _tw._topk,
        _p_d_sample_unfinished, _p_d_curandstate, _tw._end_id);
  } else {
    ker_topp_sample_launcher<_DataType>(
        _batch_size, (_cur_step + 1), _tw._max_step, 1, _max_thread_per_block,
        _stream, _p_d_logit_buf, _p_device_emb[6], _p_d_alive_seq,
        _p_d_alive_seq_buf, _tw._trg_vocab_size, _tw._topp,
        _p_d_sample_unfinished, _p_d_curandstate, _tw._end_id);
  }
#ifdef DEBUG_RESULT
  print_vec(_p_d_sample_unfinished, "unfinished flag", 1);
  for (int ii = 0; ii < _batch_size; ii++) {
    print_vec(_p_d_alive_seq + ii * _tw._max_step,
              "Batch token ids: ", _cur_step + 2);
  }
#endif

  CHECK_GPU_ERROR(cudaMemcpyAsync(&_h_unfinished, _p_d_sample_unfinished,
                                  sizeof(int), cudaMemcpyDeviceToHost,
                                  _stream));
  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));

  return _h_unfinished == 1 ? false : true;
}

template <OperationType OpType_>
bool QuantDecoder<OpType_>::beam_search() {
  /*
    step 1. logits bias and softmax,
      select rough topk candidate for every batch item,
      record the candidate's beam_id, vocab_id and probability
  */
  update_new_seq_probs();

  /* ---step 2. sort the candidate with their probability--- */
  CHECK_GPU_ERROR(cudaMemcpyAsync(&_h_can_num_batch, _p_d_can_num, sizeof(int),
                                  cudaMemcpyDeviceToHost, _stream));
  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  if (_tw._diverse_lambda != 0) {
    thrust::sort_by_key(thrust::cuda::par.on(_stream), _p_d_can_score,
                        _p_d_can_score + _h_can_num_batch, _p_d_can_idx,
                        thrust::greater<float>());

    ker_diverse_beam_search_launcher(_p_d_can_score, _p_d_can_idx, _p_d_can_num,
                                     _step_token_num, _max_thread_per_block,
                                     _stream, _tw._beam_size,
                                     _tw._diverse_lambda, _tw._trg_vocab_size);
  }

  thrust::sort_by_key(thrust::cuda::par.on(_stream), _p_d_can_score,
                      _p_d_can_score + _h_can_num_batch, _p_d_can_idx,
                      thrust::greater<float>());

#ifdef DEBUG_RESULT
  print_vec(_p_d_can_score, "can score", _h_can_num_batch);
  print_vec(_p_d_can_idx, "can idx", _h_can_num_batch);
#endif

  /*
    step 3. refresh alive_seq, seq_probs, seq_score, num_finish_beam
      based on sorted candidate.
      Deciding whether early stop based on num_finish_beam
  */
  CHECK_GPU_ERROR(cudaMemsetAsync(_p_d_can_num, 0, sizeof(int), _stream));
  ker_refresh_result<<<dim3(_batch_size, _tw._beam_size), _tw._max_step, 0,
                       _stream>>>(
      _p_d_can_idx, _p_d_can_score, _p_d_can_num + 1, _p_d_alive_seq,
      _p_d_alive_seq_buf, _p_d_alive_seq_probs, _p_d_alive_seq_score,
      _p_d_can_num, _tw._trg_vocab_size, _cur_step, _h_length_norm[_cur_step],
      _tw._diverse_lambda, _tw._end_id);
  int* tmp = _p_d_alive_seq_buf;
  _p_d_alive_seq_buf = _p_d_alive_seq;
  _p_d_alive_seq = tmp;
  CHECK_GPU_ERROR(cudaMemcpyAsync(&_h_can_num_batch, _p_d_can_num, sizeof(int),
                                  cudaMemcpyDeviceToHost, _stream));
  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));

#ifdef DEBUG_RESULT
  for (int ii = 0; ii < _batch_size; ii++) {
    for (int jj = 0; jj < _tw._beam_size; jj++) {
      print_vec(_p_d_alive_seq + (ii * _tw._beam_size + jj) * _tw._max_step,
                "Batch token ids: ", _cur_step + 2);
      print_vec(_p_d_alive_seq_probs + ii * _tw._beam_size + jj,
                "Batch probs: ", 1);
      print_vec(_p_d_alive_seq_score + ii * _tw._beam_size + jj,
                "Batch scores: ", 1);
    }
  }
#endif

  if (_h_can_num_batch == _step_token_num) {
#ifdef DEBUG_RESULT
    std::cout << "early stop beam search!" << std::endl;
#endif
    return true;
  }

  /* ---step 4. refresh cache: k, v for decoder self attention--- */
  if (_cur_step > 0) {
    ker_refresh_cache_launcher<int8_t>(
        _tw._n_dec_layer, _step_token_num * 2, _max_thread_per_block, _stream,
        _p_d_can_num + 1, _p_d_can_idx, _p_d_self_k_cache1[0],
        _p_d_self_v_cache1[0], _p_d_self_k_cache2[0], _p_d_self_v_cache2[0],
        _layer_size_self_k, _tw._beam_size, _tw._dim_per_head, _tw._head_num,
        _tw._trg_vocab_size, _cur_step, _tw._max_step, _tw._diverse_lambda != 0,
        _tw._end_id);
    int8_t** ftmp = _p_d_self_k_cache2;
    _p_d_self_k_cache2 = _p_d_self_k_cache1;
    _p_d_self_k_cache1 = ftmp;
    ftmp = _p_d_self_v_cache2;
    _p_d_self_v_cache2 = _p_d_self_v_cache1;
    _p_d_self_v_cache1 = ftmp;
  }
  return false;
}

/**
Logits bias and softmax.
Select rough topk candidate for every batch item.
Record the candidate's beam_id, vocab_id and probability
*/
template <OperationType OpType_>
void QuantDecoder<OpType_>::update_new_seq_probs() {
  CHECK_GPU_ERROR(cudaMemsetAsync(_p_d_can_num, 0, sizeof(int), _stream));

  select_beam_rough_topk_i8I_launcher(
      _int8_ffn_out_buf, _p_device_emb[6], _p_d_alive_seq_probs,
      _p_d_alive_seq_score, _p_d_alive_seq, _logits_clip_max / _quant_range,
      _p_d_can_idx, _p_d_can_score, _p_d_can_num, _tw._trg_vocab_size,
      _tw._max_step, _h_length_norm[_cur_step], _cur_step, _step_token_num,
      _max_thread_per_block, _stream, _tw._beam_size, _tw._diverse_lambda,
      _tw._end_id, true);

  thrust::exclusive_scan(thrust::cuda::par.on(_stream), _p_d_can_num + 1,
                         _p_d_can_num + 1 + _step_token_num, _p_d_can_num + 1);
  return;
}

template <OperationType OpType_>
bool QuantDecoder<OpType_>::topk_greedy_search() {
  throw std::runtime_error("QuantDecoder topk_greedy_search() not implemented");
  _tw._diverse_lambda = 0;
  if (_cur_step == 0) {
    return beam_search();
  }

  CHECK_GPU_ERROR(
      cudaMemsetAsync(_p_d_sample_unfinished, 0, sizeof(int), _stream));
  /* --- Sample new tokens from logits --- */
  ker_topk_sample_launcher<_DataType>(
      _step_token_num, (_cur_step + 1), _tw._max_step, 1, _max_thread_per_block,
      _stream, _p_d_logit_buf, _p_device_emb[6], _p_d_alive_seq,
      _p_d_alive_seq_buf, _tw._trg_vocab_size, 1, _p_d_sample_unfinished,
      _p_d_curandstate, _tw._end_id);

#ifdef DEBUG_RESULT
  print_vec(_p_d_sample_unfinished, "unfinished flag", 1);
  for (int ii = 0; ii < _batch_size; ii++) {
    for (int jj = 0; jj < _tw._beam_size; jj++) {
      print_vec(_p_d_alive_seq + (ii * _tw._beam_size + jj) * _tw._max_step,
                "Batch token ids: ", _cur_step + 2);
    }
  }
#endif

  CHECK_GPU_ERROR(cudaMemcpyAsync(&_h_unfinished, _p_d_sample_unfinished,
                                  sizeof(int), cudaMemcpyDeviceToHost,
                                  _stream));

  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));

  return _h_unfinished == 1 ? false : true;
}

template class QuantDecoder<OperationType::FP16>;
template class QuantDecoder<OperationType::FP32>;

}  // namespace cuda
}  // namespace lightseq
