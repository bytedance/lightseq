#include "../kernels/gptKernels.h"
#include "../kernels/gptKernels_int8.h"
#include "../kernels/transformerKernels.h"
#include "../kernels/transformerKernels_int8.h"
#include "quant_gpt_encoder.h"
#include "cublas_helper.h"

/**
@file
QuantGPT encoder, composed by gemm lib and
  custom cuda kernel function
*/

// #define DEBUG_RESULT

namespace lightseq {
namespace cuda {

template <OperationType OpType_>
QuantGptEncoder<OpType_>::QuantGptEncoder(
    int max_batch_size, const int *p_d_token_id, float *p_d_ppl,
    int *p_d_sample_id, const QuantGptWeight<OpType_> &tw, cudaStream_t stream,
    cudaStream_t cache_stream, cublasHandle_t hd)
    : _max_batch_size(max_batch_size),
      _p_d_token_id(p_d_token_id),
      _p_d_ppl(p_d_ppl),
      _p_d_sample_id(p_d_sample_id),
      _tw(tw),
      _stream(stream),
      _cache_stream(cache_stream),
      _hd(hd),
      _p_d_src_emb_wei(tw.get_src_emb_wei()),
      _p_d_enc_wei(tw.get_enc_wei()),
      _fone((_DataType)1.f),
      _fzero((_DataType)0.f),
      _src_emb_clip_max(tw.get_src_emb_clip_max()),
      _output_ln_clip_max(tw.get_output_ln_clip_max()),
      _logits_clip_max(tw.get_logits_clip_max()),
      _enc_clip_max(tw.get_enc_clip_max()),
      _ione((int32_t)1),
      _izero((int32_t)0),
      _atten_scaler((_DataType)sqrt(1.f / tw._dim_per_head)),
      _max_batch_dim(max_batch_size * tw._max_step * tw._hidden_size),
      _max_thread_per_block(1024),
      _h_real_seq_len(max_batch_size, 0),
      _h_ppl(max_batch_size, 0.f),
      _h_sample_id(max_batch_size * tw._max_step, 0),
      _h_unfinished(1) {
  CHECK_GPU_ERROR(cublasLtCreate(&_cublas_lt_handle));
}

/**
Init the GPU memory pointer which point to
  the memory buffer needed by encoder.
These buffer are used during custom cuda kernel function,
  find the corresponding function to see how these buffer are used
*/
template <OperationType OpType_>
void QuantGptEncoder<OpType_>::init_buffer() {
  CHECK_GPU_ERROR(
      cudaMalloc(&_p_d_real_seq_len, _max_batch_size * sizeof(int)));
  CHECK_GPU_ERROR(cudaMalloc(&_p_d_query, _max_batch_dim * sizeof(_DataType)));
  CHECK_GPU_ERROR(cudaMalloc((void **)&_p_d_curandstate,
                             _max_batch_size * sizeof(curandState)));
  CHECK_GPU_ERROR(cudaMalloc((void **)&_p_d_sample_id_buf,
                             _max_batch_size * _tw._max_step * sizeof(int)));
  CHECK_GPU_ERROR(cudaMalloc((void **)&_p_d_unfinished, sizeof(int)));
  ker_curand_setup<<<_max_batch_size, 1, 0, _stream>>>(_p_d_curandstate);

  _DataType *qkv_buf;
  CHECK_GPU_ERROR(cudaMalloc(&qkv_buf, 3 * _max_batch_dim * sizeof(_DataType)));
  _p_d_q = qkv_buf;
  _p_d_k = qkv_buf + _max_batch_dim;
  _p_d_v = qkv_buf + 2 * _max_batch_dim;

  int max_attn_score_dim = round_up(
      _max_batch_size * _tw._head_num * _tw._max_step * _tw._max_step, 32);

  CHECK_GPU_ERROR(cudaMalloc(&_p_d_c, max_attn_score_dim * sizeof(_DataType)));

  int max_batch_dim =
      _max_batch_size * _tw._max_step *
      round_up(std::max(_tw._inner_size, _tw._hidden_size * 3), 32);
  CHECK_GPU_ERROR(
      cudaMalloc(&_int8_ffn_in_buf, max_batch_dim * sizeof(int8_t)));
  CHECK_GPU_ERROR(cudaMalloc(
      &_int32_ffn_out_buf,
      std::max(max_batch_dim, max_attn_score_dim) * sizeof(int32_t)));
  CHECK_GPU_ERROR(
      cudaMalloc(&_int8_ffn_out_buf,
                 std::max(max_batch_dim, round_up(_tw._src_vocab_size, 32) *
                                             _tw._max_step * _max_batch_size) *
                     sizeof(int8_t)));

  // malloc embeddings
  CHECK_GPU_ERROR(
      cudaMalloc(&_int8_p_d_src_emb_wei,
                 _tw._src_vocab_size * _tw._hidden_size * sizeof(int8_t)));
  quantize_weight(_p_d_src_emb_wei[0], _int8_p_d_src_emb_wei, _tw._hidden_size,
                  _tw._src_vocab_size, _quant_range / _src_emb_clip_max,
                  _stream, _cublas_lt_handle);
  CHECK_GPU_ERROR(
      cudaMalloc(&_int8_p_d_src_emb_bottom_wei,
                 _tw._src_vocab_size * _tw._hidden_size * sizeof(int8_t)));
  quantize_weight(_p_d_src_emb_wei[0], _int8_p_d_src_emb_bottom_wei,
                  _tw._hidden_size, _tw._src_vocab_size,
                  _quant_range / _src_emb_clip_max, _stream, _cublas_lt_handle,
                  kColMajor);
  _p_device_emb.push_back(nullptr);
  _p_device_emb.push_back(
      to_gpu(_p_d_src_emb_wei[1], _tw._max_step * _tw._hidden_size, _stream));
  _p_device_emb.push_back(
      to_gpu(_p_d_src_emb_wei[2], _tw._hidden_size, _stream));
  _p_device_emb.push_back(
      to_gpu(_p_d_src_emb_wei[3], _tw._hidden_size, _stream));

  // malloc reused kv cache max size: _tw._hidden_size * 2 * _tw._n_enc_layer *
  // _max_batch_size * _max_step * sizeof(T)
  int8_t *self_kv_cache_buffer;
  int8_t *sliding_p;
  CHECK_GPU_ERROR(
      cudaMalloc(&self_kv_cache_buffer,
                 _max_batch_dim * _tw._n_enc_layer * 4 * sizeof(int8_t)));

  sliding_p = self_kv_cache_buffer;
  for (int i = 0; i < _tw._n_enc_layer * 2; i++) {
    _p_d_self_k_cache.push_back(sliding_p);
    sliding_p += _max_batch_dim;
  }
  for (int i = 0; i < _tw._n_enc_layer * 2; i++) {
    _p_d_self_v_cache.push_back(sliding_p);
    sliding_p += _max_batch_dim;
  }
  _p_d_self_k_cache1 = _p_d_self_k_cache.data();
  _p_d_self_k_cache2 = _p_d_self_k_cache.data() + _tw._n_enc_layer;
  _p_d_self_v_cache1 = _p_d_self_v_cache.data();
  _p_d_self_v_cache2 = _p_d_self_v_cache.data() + _tw._n_enc_layer;

  // malloc weights
  _int8_p_d_enc_wei = std::vector<int8_t *>(_tw._n_enc_layer * 4);
  _scaled_ffn2_colsum = std::vector<_DataType *>(_tw._n_enc_layer);
  for (_layer_id = 0; _layer_id < _tw._n_enc_layer; _layer_id++) {
    _weight_offset = _layer_id * _tw._weight_per_enc_layer;
    // malloc quantized weights
    CHECK_GPU_ERROR(
        cudaMalloc(&_int8_p_d_enc_wei[_layer_id * 4],
                   _tw._hidden_size * 3 * _tw._hidden_size * sizeof(int8_t)));
    CHECK_GPU_ERROR(
        cudaMalloc(&_int8_p_d_enc_wei[_layer_id * 4 + 1],
                   _tw._hidden_size * _tw._hidden_size * sizeof(int8_t)));
    CHECK_GPU_ERROR(
        cudaMalloc(&_int8_p_d_enc_wei[_layer_id * 4 + 2],
                   _tw._hidden_size * _tw._inner_size * sizeof(int8_t)));
    CHECK_GPU_ERROR(
        cudaMalloc(&_int8_p_d_enc_wei[_layer_id * 4 + 3],
                   _tw._inner_size * _tw._hidden_size * sizeof(int8_t)));

    // malloc unquantized weights
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
                    _quant_range / _enc_clip_max[_layer_id * 12], _stream,
                    _cublas_lt_handle);

    quantize_weight(_p_d_enc_wei[_weight_offset + 4],
                    _int8_p_d_enc_wei[_layer_id * 4 + 1], _tw._hidden_size,
                    _tw._hidden_size,
                    _quant_range / _enc_clip_max[_layer_id * 12 + 1], _stream,
                    _cublas_lt_handle, kColMajor);

    quantize_weight(_p_d_enc_wei[_weight_offset + 8],
                    _int8_p_d_enc_wei[_layer_id * 4 + 2], _tw._hidden_size,
                    _tw._inner_size,
                    _quant_range / _enc_clip_max[_layer_id * 12 + 2], _stream,
                    _cublas_lt_handle);

    quantize_weight(_p_d_enc_wei[_weight_offset + 10],
                    _int8_p_d_enc_wei[_layer_id * 4 + 3], _tw._inner_size,
                    _tw._hidden_size,
                    _quant_range / _enc_clip_max[_layer_id * 12 + 3], _stream,
                    _cublas_lt_handle, kColMajor);

    _scaled_ffn2_colsum[_layer_id] = nullptr;
  }

  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  CHECK_GPU_ERROR(cudaGetLastError());
  std::cout << "quantized encoder buffer init succeed" << std::endl;

  return;
}

/**
Some requirements needed by custom cuda kernel function
*/
template <OperationType OpType_>
std::string QuantGptEncoder<OpType_>::check() {
  // if (_max_thread_per_block < _tw._hidden_size) {
  //   return "violate hidden_size <= max_thread_per_block";
  // }
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
  std::string sampling_method = _tw._sampling_method;
  if (kSamplingMethods.find(sampling_method) == kSamplingMethods.end()) {
    return std::string("unsupported sampling_method: ") + sampling_method;
  }

  if (_tw._topk <= 0) {
    return "topk must be positive";
  }
  if (_tw._topp <= 0 && _tw._topp >= 1.0) {
    return "topp must be in (0, 1)";
  }

  return "";
}

template <OperationType OpType_>
void QuantGptEncoder<OpType_>::run_one_infer(int batch_size,
                                             int batch_seq_len) {
  if (batch_size > _max_batch_size) {
    throw std::runtime_error("batch size of input greater than max_batch_size");
  }
  if (batch_seq_len > _tw._max_step) {
    throw std::runtime_error("seq len of input greater than max_step");
  }
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
  std::cout << "batch_size-" << batch_size << " batch_seq_len-" << batch_seq_len
            << std::endl;
  print_vec(_p_d_token_id, "batch_token_ids", batch_size * batch_seq_len);
#endif

  // token embedding, add position embedding and layer_norm
  ker_gpt_embedding_i8I_launcher<_DataType>(
      batch_size, batch_seq_len, _tw._hidden_size, _stream,
      _int8_p_d_src_emb_bottom_wei, _p_device_emb[1], _p_d_token_id, _p_d_query,
      _p_d_real_seq_len, _tw._padding_id, 0, _src_emb_clip_max / _quant_range);

  for (_layer_id = 0; _layer_id < _tw._n_enc_layer; _layer_id++) {
    _weight_offset = _layer_id * _tw._weight_per_enc_layer;
    self_attention();
    ffn_add_norm();
  }

  compute_ppl();
  return;
}

template <OperationType OpType_>
int QuantGptEncoder<OpType_>::run_one_sample(int batch_size,
                                             int batch_seq_len) {
  if (batch_size > _max_batch_size) {
    throw std::runtime_error("batch size of input greater than max_batch_size");
  }
  if (batch_seq_len > _tw._max_step) {
    throw std::runtime_error("seq len of input greater than max_step");
  }
  _batch_size = batch_size;
  _batch_seq_len = batch_seq_len;
  _batch_token_num = batch_size * batch_seq_len;

  CHECK_GPU_ERROR(cudaMemcpyAsync(_p_d_real_seq_len, _h_real_seq_len.data(),
                                  sizeof(int) * _batch_size,
                                  cudaMemcpyHostToDevice, _stream));
  CHECK_GPU_ERROR(cudaMemcpyAsync(_p_d_ppl, _h_ppl.data(),
                                  sizeof(float) * _batch_size,
                                  cudaMemcpyHostToDevice, _stream));
  CHECK_GPU_ERROR(cudaMemcpyAsync(_p_d_sample_id, _p_d_token_id,
                                  sizeof(int) * _batch_size * _batch_seq_len,
                                  cudaMemcpyDeviceToDevice, _stream));
#ifdef DEBUG_RESULT
  std::cout << "batch_size-" << batch_size << " batch_seq_len-" << batch_seq_len
            << std::endl;
  std::cout << "Sample with " << _tw._sampling_method << std::endl;
  std::cout << "padding_id: " << _tw._padding_id << std::endl;
  std::cout << "vocab_size: " << _tw._src_vocab_size << std::endl;
  print_vec(_p_d_sample_id, "batch_token_ids", batch_size * batch_seq_len);
#endif

  // token embedding, add position embedding and layer_norm
  ker_gpt_embedding_i8I_launcher<_DataType>(
      _batch_size, _batch_seq_len, _tw._hidden_size, _stream,
      _int8_p_d_src_emb_bottom_wei, _p_device_emb[1], _p_d_sample_id,
      _p_d_query, _p_d_real_seq_len, _tw._padding_id, 0,
      _src_emb_clip_max / _quant_range);

  for (_layer_id = 0; _layer_id < _tw._n_enc_layer; _layer_id++) {
    _weight_offset = _layer_id * _tw._weight_per_enc_layer;
    self_attention();
    ffn_add_norm();
  }

  int8_t **ftmp = _p_d_self_k_cache2;
  _p_d_self_k_cache2 = _p_d_self_k_cache1;
  _p_d_self_k_cache1 = ftmp;
  ftmp = _p_d_self_v_cache2;
  _p_d_self_v_cache2 = _p_d_self_v_cache1;
  _p_d_self_v_cache1 = ftmp;

  if (sample_one_token() == 0 || _batch_seq_len >= _tw._max_step) {
    CHECK_GPU_ERROR(cudaMemcpyAsync(_p_d_sample_id_buf, _p_d_sample_id,
                                    _batch_token_num * sizeof(int),
                                    cudaMemcpyDeviceToDevice, _stream));
    CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
    return _batch_seq_len;
  }

  while (1) {
#ifdef DEBUG_RESULT
    std::cout << "before sample:batch_size-" << _batch_size << " batch_seq_len-"
              << _batch_seq_len << std::endl;
    print_vec(_p_d_sample_id, "batch_token_ids", _batch_token_num);
#endif

    // token embedding, add position embedding and layer_norm
    ker_gpt_embedding_i8I_launcher<_DataType>(
        batch_size, 1, _tw._hidden_size, _stream, _int8_p_d_src_emb_bottom_wei,
        _p_device_emb[1], _p_d_last_sample_id, _p_d_query, _p_d_real_seq_len,
        _tw._padding_id, _batch_seq_len - 1, _src_emb_clip_max / _quant_range);

    for (_layer_id = 0; _layer_id < _tw._n_enc_layer; _layer_id++) {
      _weight_offset = _layer_id * _tw._weight_per_enc_layer;
      self_attention_with_cache();
      ffn_add_norm_with_cache();
    }

    int8_t **ftmp = _p_d_self_k_cache2;
    _p_d_self_k_cache2 = _p_d_self_k_cache1;
    _p_d_self_k_cache1 = ftmp;
    ftmp = _p_d_self_v_cache2;
    _p_d_self_v_cache2 = _p_d_self_v_cache1;
    _p_d_self_v_cache1 = ftmp;

    if (sample_one_token_with_cache() == 0 || _batch_seq_len >= _tw._max_step)
      break;
  }

  CHECK_GPU_ERROR(cudaMemcpyAsync(_p_d_sample_id_buf, _p_d_sample_id,
                                  _batch_token_num * sizeof(int),
                                  cudaMemcpyDeviceToDevice, _stream));
  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));

  return _batch_seq_len;
}

template <OperationType OpType_>
int QuantGptEncoder<OpType_>::sample_one_token() {
  /* ---step 1. project hidden states to vocab logits--- */
  cublasLtMM_withAlgo_i8IO(_int8_ffn_out_buf, 1, _batch_token_num,
                           _tw._src_vocab_size, _tw._hidden_size, 0, 0, 0,
                           _output_ln_clip_max * _src_emb_clip_max /
                               (_logits_clip_max * _quant_range),
                           _int8_ffn_in_buf, _int8_p_d_src_emb_wei,
                           _cublas_lt_handle, _stream, use_ORDER_COL32_2R_4R4);
  CHECK_GPU_ERROR(cudaMemsetAsync(_p_d_unfinished, 0, sizeof(int), _stream));
  /* ---step 2. sample new tokens from logits */
  if (_tw._sampling_method == "topk") {
#ifdef DEBUG_RESULT
    std::cout << "sampling using topk\n";
#endif
    ker_topk_sample_i8I_launcher(
        _batch_size, _batch_seq_len, _batch_seq_len, _max_thread_per_block,
        _stream, _int8_ffn_out_buf, _p_d_sample_id, _p_d_sample_id_buf,
        _p_d_real_seq_len, _tw._src_vocab_size, _tw._topk, _p_d_unfinished,
        _p_d_curandstate, _tw._eos_id, _logits_clip_max / _quant_range, true);
  } else {
#ifdef DEBUG_RESULT
    std::cout << "sampling using topp\n";
#endif
    ker_topp_sample_i8I_launcher(
        _batch_size, _batch_seq_len, _batch_seq_len, _max_thread_per_block,
        _stream, _int8_ffn_out_buf, _p_d_sample_id, _p_d_sample_id_buf,
        _p_d_real_seq_len, _tw._src_vocab_size, _tw._topp, _p_d_unfinished,
        _p_d_curandstate, _tw._eos_id, _logits_clip_max / _quant_range, true);
  }
  int *temp = _p_d_sample_id;
  _p_d_sample_id = _p_d_sample_id_buf;
  _p_d_sample_id_buf = temp;
  CHECK_GPU_ERROR(cudaMemcpyAsync(&_h_unfinished, _p_d_unfinished, sizeof(int),
                                  cudaMemcpyDeviceToHost, _stream));
  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  _p_d_last_sample_id = _p_d_sample_id_buf + _batch_token_num;
  _batch_seq_len++;
  _batch_token_num += _batch_size;
  return _h_unfinished;
}

template <OperationType OpType_>
int QuantGptEncoder<OpType_>::sample_one_token_with_cache() {
  /* ---step 1. project hidden states to vocab logits--- */
  cublasLtMM_withAlgo_i8IO(_int8_ffn_out_buf, 1, _batch_size,
                           _tw._src_vocab_size, _tw._hidden_size, 0, 0, 0,
                           _output_ln_clip_max * _src_emb_clip_max /
                               (_logits_clip_max * _quant_range),
                           _int8_ffn_in_buf, _int8_p_d_src_emb_wei,
                           _cublas_lt_handle, _stream, use_ORDER_COL32_2R_4R4);

  CHECK_GPU_ERROR(cudaMemsetAsync(_p_d_unfinished, 0, sizeof(int), _stream));
  // /* ---step 2. sample new tokens from logits */
  if (_tw._sampling_method == "topk") {
#ifdef DEBUG_RESULT
    std::cout << "sampling using topk\n";
#endif
    ker_topk_sample_i8I_launcher(
        _batch_size, _batch_seq_len, 1, _max_thread_per_block, _stream,
        _int8_ffn_out_buf, _p_d_sample_id, _p_d_sample_id_buf,
        _p_d_real_seq_len, _tw._src_vocab_size, _tw._topk, _p_d_unfinished,
        _p_d_curandstate, _tw._eos_id, _logits_clip_max / _quant_range, true);
  } else {
#ifdef DEBUG_RESULT
    std::cout << "sampling using topp\n";
#endif
    ker_topp_sample_i8I_launcher(
        _batch_size, _batch_seq_len, 1, _max_thread_per_block, _stream,
        _int8_ffn_out_buf, _p_d_sample_id, _p_d_sample_id_buf,
        _p_d_real_seq_len, _tw._src_vocab_size, _tw._topp, _p_d_unfinished,
        _p_d_curandstate, _tw._eos_id, _logits_clip_max / _quant_range, true);
  }
  int *temp = _p_d_sample_id;
  _p_d_sample_id = _p_d_sample_id_buf;
  _p_d_sample_id_buf = temp;
  CHECK_GPU_ERROR(cudaMemcpyAsync(&_h_unfinished, _p_d_unfinished, sizeof(int),
                                  cudaMemcpyDeviceToHost, _stream));
  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  _p_d_last_sample_id = _p_d_sample_id_buf + _batch_token_num;
  _batch_seq_len++;
  _batch_token_num += _batch_size;
  return _h_unfinished;
}

template <OperationType OpType_>
void QuantGptEncoder<OpType_>::self_attention() {
  /* ---step 0. layer_norm, add output_bias to "query"--- */
  if (_layer_id == 0) {
    ker_norm_layer_resual_i8O_launcher<_DataType>(
        _batch_token_num, _tw._hidden_size, _stream, _p_d_query,
        _int8_ffn_in_buf, _p_device_wei[_weight_offset],
        _p_device_wei[_weight_offset + 1], _p_device_wei[_weight_offset + 5],
        _max_thread_per_block, _quant_range / _enc_clip_max[_layer_id * 12 + 4],
        false, true);
  }

  cublasLtMM_withAlgo_i8IO(
      _int8_ffn_out_buf, 1, _batch_token_num, _tw._hidden_size * 3,
      _tw._hidden_size, 0, 0, 0,
      _enc_clip_max[_layer_id * 12] * _enc_clip_max[_layer_id * 12 + 4] /
          (_enc_clip_max[_layer_id * 12 + 8] * _quant_range),
      _int8_ffn_in_buf, _int8_p_d_enc_wei[_layer_id * 4], _cublas_lt_handle,
      _stream, use_ORDER_COL32_2R_4R4);

#ifdef DEBUG_RESULT
  print_vec(_int8_ffn_in_buf, "attn qkv in", 20);
  print_vec(_int8_p_d_enc_wei[_layer_id * 4], "attn qkv w", 20);
  print_vec(_int8_ffn_out_buf, "attn qkv out", 20);
#endif

  // get q, k, v by split and reshape qkv
  ker_arrange_encself_qkv_i8I_i8O_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _int8_ffn_out_buf,
      _p_device_wei[_weight_offset + 3], _int8_ffn_in_buf,
      _p_d_self_k_cache1[_layer_id], _p_d_self_v_cache1[_layer_id], _p_d_v,
      _batch_seq_len, _tw._dim_per_head, _tw._head_num, _max_thread_per_block,
      _enc_clip_max[_layer_id * 12 + 8] / _quant_range,
      _quant_range / _enc_clip_max[_layer_id * 12 + 11], true);

  /* ---step 2. correlation = q * k, perform softmax on correlation--- */
  CHECK_GPU_ERROR(cublasGemmStridedBatchedEx(
      _hd, CUBLAS_OP_T, CUBLAS_OP_N, _batch_seq_len, _batch_seq_len,
      _tw._dim_per_head, &_ione, _p_d_self_k_cache1[_layer_id], CUDA_R_8I,
      _tw._dim_per_head, _batch_seq_len * _tw._dim_per_head, _int8_ffn_in_buf,
      CUDA_R_8I, _tw._dim_per_head, _batch_seq_len * _tw._dim_per_head, &_izero,
      _int32_ffn_out_buf, CUDA_R_32I, _batch_seq_len,
      _batch_seq_len * _batch_seq_len, _batch_size * _tw._head_num, CUDA_R_32I,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  ker_correlation_softmax_gpt_i32I_launcher<_DataType>(
      _batch_size, _batch_seq_len, _tw._head_num, _stream, _int32_ffn_out_buf,
      _p_d_c, _p_d_real_seq_len, _atten_scaler,
      _enc_clip_max[_layer_id * 12 + 11] / _quant_range);

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
      _quant_range / _enc_clip_max[_layer_id * 12 + 5], false);

  /* ---step 4. new_q = ori_q + new_q * output_wei--- */
  cublaslt_gemm(
      _int8_p_d_enc_wei[_layer_id * 4 + 1], _int8_ffn_in_buf, _int8_ffn_out_buf,
      1, _tw._hidden_size, _batch_token_num, _tw._hidden_size, 0, 0, 0,
      _enc_clip_max[_layer_id * 12 + 1] * _enc_clip_max[_layer_id * 12 + 5] /
          (_enc_clip_max[_layer_id * 12 + 9] * _quant_range),
      _cublas_lt_handle, _stream);

#ifdef DEBUG_RESULT
  print_vec(_int8_ffn_in_buf, "attn out in", 20);
  print_vec(_int8_p_d_enc_wei[_layer_id * 4 + 1], "attn out w", 20);
  print_vec(_int8_ffn_out_buf, "attn out out", 20);
#endif

  ker_residual_bias_ln_i8I_i8O_launcher<_DataType>(
      _int8_ffn_out_buf, _p_device_wei[_weight_offset + 6],
      _p_device_wei[_weight_offset + 7], _p_device_wei[_weight_offset + 11],
      _int8_ffn_in_buf, _p_d_query, _batch_token_num, _tw._hidden_size,
      _enc_clip_max[_layer_id * 12 + 9] / _quant_range,
      _quant_range / _enc_clip_max[_layer_id * 12 + 6], _max_thread_per_block,
      _stream, false, false, true);

  return;
}

template <OperationType OpType_>
void QuantGptEncoder<OpType_>::self_attention_with_cache() {
  /* ---step 0. layer_norm, add output_bias to "query"--- */
  if (_layer_id == 0) {
    ker_norm_layer_resual_i8O_launcher<_DataType>(
        _batch_size, _tw._hidden_size, _stream, _p_d_query, _int8_ffn_in_buf,
        _p_device_wei[_weight_offset], _p_device_wei[_weight_offset + 1],
        _p_device_wei[_weight_offset + 5], _max_thread_per_block,
        _quant_range / _enc_clip_max[_layer_id * 12 + 4], false, true);
  }

  /* ---step 1. qkv = ori_q * qkv_wei + bias, and reshape qkv for multi-head
   * gemm--- */
  cublasLtMM_withAlgo_i8IO(
      _int8_ffn_out_buf, 1, _batch_size, _tw._hidden_size * 3, _tw._hidden_size,
      0, 0, 0,
      _enc_clip_max[_layer_id * 12] * _enc_clip_max[_layer_id * 12 + 4] /
          (_enc_clip_max[_layer_id * 12 + 8] * _quant_range),
      _int8_ffn_in_buf, _int8_p_d_enc_wei[_layer_id * 4], _cublas_lt_handle,
      _stream, use_ORDER_COL32_2R_4R4);

  // get q, k, v by split and reshape qkv
  ker_arrange_qkv_with_cache_i8I_i8O_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _int8_ffn_out_buf,
      _p_device_wei[_weight_offset + 3], _int8_ffn_in_buf,
      _p_d_self_k_cache1[_layer_id], _p_d_self_k_cache2[_layer_id],
      _p_d_self_v_cache1[_layer_id], _p_d_self_v_cache2[_layer_id],
      _batch_seq_len, _tw._dim_per_head, _tw._head_num,
      _enc_clip_max[_layer_id * 12 + 8] / _quant_range,
      _quant_range / _enc_clip_max[_layer_id * 12 + 11], true);

  /* ---step 2. correlation = q * k, perform softmax on correlation
  correlation: [batch_size, heads_num, 1, batch_seq_len]--- */
  CHECK_GPU_ERROR(cublasGemmStridedBatchedEx(
      _hd, CUBLAS_OP_T, CUBLAS_OP_N, _batch_seq_len, 1, _tw._dim_per_head,
      &_ione, _p_d_self_k_cache1[_layer_id], CUDA_R_8I, _tw._dim_per_head,
      _batch_seq_len * _tw._dim_per_head, _int8_ffn_in_buf, CUDA_R_8I,
      _tw._dim_per_head, _tw._dim_per_head, &_izero, _int32_ffn_out_buf,
      CUDA_R_32I, _batch_seq_len, _batch_seq_len, _batch_size * _tw._head_num,
      CUDA_R_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  ker_fuse_softmax_new_value_i32I_i8O_launcher(
      _int32_ffn_out_buf, _p_d_self_v_cache1[_layer_id], _int8_ffn_in_buf,
      _batch_size * _tw._head_num, _batch_seq_len, _batch_seq_len,
      _tw._head_num, _tw._dim_per_head, float(_atten_scaler),
      _enc_clip_max[_layer_id * 12 + 11] / _quant_range,
      _quant_range / _enc_clip_max[_layer_id * 12 + 5], false, _stream);

  /* ---step 4. new_q = ori_q + new_q * output_wei--- */
  cublaslt_gemm(
      _int8_p_d_enc_wei[_layer_id * 4 + 1], _int8_ffn_in_buf, _int8_ffn_out_buf,
      1, _tw._hidden_size, _batch_size, _tw._hidden_size, 0, 0, 0,
      _enc_clip_max[_layer_id * 12 + 1] * _enc_clip_max[_layer_id * 12 + 5] /
          (_enc_clip_max[_layer_id * 12 + 9] * _quant_range),
      _cublas_lt_handle, _stream);

  ker_residual_bias_ln_i8I_i8O_launcher<_DataType>(
      _int8_ffn_out_buf, _p_device_wei[_weight_offset + 6],
      _p_device_wei[_weight_offset + 7], _p_device_wei[_weight_offset + 11],
      _int8_ffn_in_buf, _p_d_query, _batch_size, _tw._hidden_size,
      _enc_clip_max[_layer_id * 12 + 9] / _quant_range,
      _quant_range / _enc_clip_max[_layer_id * 12 + 6], _max_thread_per_block,
      _stream, false, false, true);
  return;
}

template <OperationType OpType_>
void QuantGptEncoder<OpType_>::ffn_add_norm() {
  /* ---step 1. first ffn layer--- */
  cublasLtMM_withAlgo_i8IO(
      _int8_ffn_out_buf, 1, _batch_token_num, _tw._inner_size, _tw._hidden_size,
      0, 0, 0,
      _enc_clip_max[_layer_id * 12 + 2] * _enc_clip_max[_layer_id * 12 + 6] /
          (_enc_clip_max[_layer_id * 12 + 10] * _quant_range),
      _int8_ffn_in_buf, _int8_p_d_enc_wei[_layer_id * 4 + 2], _cublas_lt_handle,
      _stream, use_ORDER_COL32_2R_4R4);

#ifdef DEBUG_RESULT
  print_vec(_int8_ffn_in_buf, "ffn1 in", 20);
  print_vec(_int8_p_d_enc_wei[_layer_id * 4 + 2], "ffn1 w", 20);
  print_vec(_int8_ffn_out_buf, "ffn1 out", 20);
#endif

  ker_bias_gelu_i8I_i8O_launcher<_DataType>(
      _batch_token_num, _stream, _int8_ffn_out_buf, _int8_ffn_in_buf,
      _p_device_wei[_weight_offset + 9], _tw._inner_size,
      _enc_clip_max[_layer_id * 12 + 10] / _quant_range,
      _quant_range / _enc_clip_max[_layer_id * 12 + 7], true, false);

  /* ---step 2. second ffn layer--- */
  cublaslt_gemm(_int8_p_d_enc_wei[_layer_id * 4 + 3], _int8_ffn_in_buf,
                _int32_ffn_out_buf, 1, _tw._hidden_size, _batch_token_num,
                _tw._inner_size, 0, 0, 0, 1, _cublas_lt_handle, _stream);

#ifdef DEBUG_RESULT
  print_vec(_int8_ffn_in_buf, "ffn2 in", 20);
  print_vec(_int8_p_d_enc_wei[_layer_id * 4 + 3], "ffn2 w", 20);
  print_vec(_int32_ffn_out_buf, "ffn2 out", 20);
#endif

  const _DataType *scale_ptr, *bias_ptr, *res_bias_ptr;
  float clip_max, dequant_scale;
  dequant_scale = _enc_clip_max[_layer_id * 12 + 3] *
                  _enc_clip_max[_layer_id * 12 + 7] /
                  (_quant_range * _quant_range);
  if (_layer_id == _tw._n_enc_layer - 1) {
    scale_ptr = _p_device_emb[2];
    bias_ptr = _p_device_emb[3];
    res_bias_ptr = nullptr;
    clip_max = _output_ln_clip_max;
  } else {
    scale_ptr = _p_device_wei[(_layer_id + 1) * _tw._weight_per_enc_layer];
    bias_ptr = _p_device_wei[(_layer_id + 1) * _tw._weight_per_enc_layer + 1];
    res_bias_ptr =
        _p_device_wei[(_layer_id + 1) * _tw._weight_per_enc_layer + 5];
    clip_max = _enc_clip_max[(_layer_id + 1) * 12 + 4];
  }

  ker_residual_bias_ln_i32I_i8O_launcher<_DataType>(
      _int32_ffn_out_buf, scale_ptr, bias_ptr, res_bias_ptr, _int8_ffn_in_buf,
      _p_d_query, _batch_token_num, _tw._hidden_size, dequant_scale,
      _quant_range / clip_max, _max_thread_per_block, _stream, false, false,
      true, _scaled_ffn2_colsum[_layer_id]);

  return;
}

template <OperationType OpType_>
void QuantGptEncoder<OpType_>::ffn_add_norm_with_cache() {
  /* ---step 1. first ffn layer--- */
  cublasLtMM_withAlgo_i8IO(
      _int8_ffn_out_buf, 1, _batch_size, _tw._inner_size, _tw._hidden_size, 0,
      0, 0,
      _enc_clip_max[_layer_id * 12 + 2] * _enc_clip_max[_layer_id * 12 + 6] /
          (_enc_clip_max[_layer_id * 12 + 10] * _quant_range),
      _int8_ffn_in_buf, _int8_p_d_enc_wei[_layer_id * 4 + 2], _cublas_lt_handle,
      _stream, use_ORDER_COL32_2R_4R4);

  ker_bias_gelu_i8I_i8O_launcher<_DataType>(
      _batch_size, _stream, _int8_ffn_out_buf, _int8_ffn_in_buf,
      _p_device_wei[_weight_offset + 9], _tw._inner_size,
      _enc_clip_max[_layer_id * 12 + 10] / _quant_range,
      _quant_range / _enc_clip_max[_layer_id * 12 + 7], true, false);

  /* ---step 2. second ffn layer--- */
  cublaslt_gemm(_int8_p_d_enc_wei[_layer_id * 4 + 3], _int8_ffn_in_buf,
                _int32_ffn_out_buf, 1, _tw._hidden_size, _batch_size,
                _tw._inner_size, 0, 0, 0, 1, _cublas_lt_handle, _stream);

  const _DataType *scale_ptr, *bias_ptr, *res_bias_ptr;
  float clip_max, dequant_scale;
  dequant_scale = _enc_clip_max[_layer_id * 12 + 3] *
                  _enc_clip_max[_layer_id * 12 + 7] /
                  (_quant_range * _quant_range);
  if (_layer_id == _tw._n_enc_layer - 1) {
    scale_ptr = _p_device_emb[2];
    bias_ptr = _p_device_emb[3];
    res_bias_ptr = nullptr;
    clip_max = _output_ln_clip_max;
  } else {
    scale_ptr = _p_device_wei[(_layer_id + 1) * _tw._weight_per_enc_layer];
    bias_ptr = _p_device_wei[(_layer_id + 1) * _tw._weight_per_enc_layer + 1];
    res_bias_ptr =
        _p_device_wei[(_layer_id + 1) * _tw._weight_per_enc_layer + 5];
    clip_max = _enc_clip_max[(_layer_id + 1) * 12 + 4];
  }

  ker_residual_bias_ln_i32I_i8O_launcher<_DataType>(
      _int32_ffn_out_buf, scale_ptr, bias_ptr, res_bias_ptr, _int8_ffn_in_buf,
      _p_d_query, _batch_size, _tw._hidden_size, dequant_scale,
      _quant_range / clip_max, _max_thread_per_block, _stream, false, false,
      true, _scaled_ffn2_colsum[_layer_id]);

  return;
}

/**
Compute ppl from encoder output
*/
template <OperationType OpType_>
void QuantGptEncoder<OpType_>::compute_ppl() {
  /* ---step 1. project hidden states to vocab logits--- */
  cublasLtMM_withAlgo_i8IO(_int8_ffn_out_buf, 1, _batch_token_num,
                           _tw._src_vocab_size, _tw._hidden_size, 0, 0, 0,
                           _output_ln_clip_max * _src_emb_clip_max /
                               (_logits_clip_max * _quant_range),
                           _int8_ffn_in_buf, _int8_p_d_src_emb_wei,
                           _cublas_lt_handle, _stream, use_ORDER_COL32_2R_4R4);
#ifdef DEBUG_RESULT
  print_vec(_int8_ffn_in_buf, "logits in", 20);
  print_vec(_int8_p_d_src_emb_wei, "logits w", 20);
  print_vec(_int8_ffn_out_buf, "logits out", 20);
#endif

  /* ---step 2. compute language model ppl--- */
  ker_ppl_i8I_launcher(_batch_size, _batch_seq_len, _max_thread_per_block,
                       _stream, _int8_ffn_out_buf, _p_d_token_id,
                       _p_d_real_seq_len, _p_d_ppl, _tw._src_vocab_size,
                       _logits_clip_max / _quant_range, true);
}

template class QuantGptEncoder<OperationType::FP16>;
template class QuantGptEncoder<OperationType::FP32>;

}  // namespace cuda
}  // namespace lightseq
