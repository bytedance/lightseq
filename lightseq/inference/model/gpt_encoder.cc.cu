#include "../kernels/gptKernels.h"
#include "../kernels/transformerKernels.h"
#include "gpt_encoder.h"

/**
@file
GPT encoder, composed by gemm lib and
  custom cuda kernel function
*/

// #define DEBUG_RESULT

namespace lightseq {
namespace cuda {

template <OperationType OpType_>
GptEncoder<OpType_>::GptEncoder(int max_batch_size, const int *p_d_token_id,
                                float *p_d_ppl, int *p_d_sample_id,
                                const GptWeight<OpType_> &tw,
                                cudaStream_t stream, cudaStream_t cache_stream,
                                cublasHandle_t hd)
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
      _atten_scaler((_DataType)sqrt(1.f / tw._dim_per_head)),
      _max_batch_dim(max_batch_size * tw._max_step * tw._hidden_size),
      _max_thread_per_block(1024),
      _h_real_seq_len(max_batch_size, 0),
      _h_ppl(max_batch_size, 0.f),
      _h_sample_id(max_batch_size * tw._max_step, 0),
      _h_unfinished(1) {}

/**
Compute GPU memory size needed by gpt encoder,
  to see how these memory is used, checkout init_buffer() for detail
*/
template <OperationType OpType_>
size_t GptEncoder<OpType_>::compute_buffer_bytesize() {
  int si = _max_batch_size;
  size_t sz0 = (size_t)_max_batch_dim;
  sz0 += 2 * (size_t)_max_batch_dim * (size_t)_tw._n_enc_layer;
  long long sz1 = (size_t)_max_batch_dim * 6 +
                  (size_t)_max_batch_size * (size_t)_tw._head_num *
                      (size_t)_tw._max_step * (size_t)_tw._max_step;
  long long sz2 = (size_t)_max_batch_dim + (size_t)_max_batch_size *
                                               (size_t)_tw._max_step *
                                               (size_t)_tw._inner_size;
  long long sz3 = (size_t)_max_batch_size * (size_t)_tw._max_step *
                  (size_t)_tw._src_vocab_size;
  return (sz0 + max(max(sz1, sz2), sz3)) * sizeof(_DataType) + si * sizeof(int);
}

/**
Init the GPU memory pointer which point to
  the memory buffer needed by encoder.
These buffer are used during custom cuda kernel function,
  find the corresponding function to see how these buffer are used
*/
template <OperationType OpType_>
void GptEncoder<OpType_>::init_buffer(void *pbuf) {
  // int buffer
  int *p_d_int = reinterpret_cast<int *>(pbuf);
  _p_d_real_seq_len = p_d_int;
  p_d_int += _max_batch_size;

  // datatype buffer
  _DataType *p_d_datatype = reinterpret_cast<_DataType *>(p_d_int);
  _p_d_query = p_d_datatype;
  _p_d_k_cache = _p_d_query + _max_batch_dim;
  _p_d_v_cache = _p_d_k_cache + _max_batch_dim * _tw._n_enc_layer;
  p_d_datatype = _p_d_v_cache + _max_batch_dim * _tw._n_enc_layer;
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
  CHECK_GPU_ERROR(cudaMalloc((void **)&_p_d_curandstate,
                             _max_batch_size * sizeof(curandState)));
  CHECK_GPU_ERROR(cudaMalloc((void **)&_p_d_sample_id_buf,
                             _max_batch_size * _tw._max_step * sizeof(int)));
  CHECK_GPU_ERROR(cudaMalloc((void **)&_p_d_unfinished, sizeof(int)));
  ker_curand_setup<<<_max_batch_size, 1, 0, _stream>>>(_p_d_curandstate);
  return;
}

/**
Some requirements needed by custom cuda kernel function
*/
template <OperationType OpType_>
std::string GptEncoder<OpType_>::check() {
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
  std::cout << "batch_size-" << batch_size << " batch_seq_len-" << batch_seq_len
            << std::endl;
  print_vec(_p_d_token_id, "batch_token_ids", batch_size * batch_seq_len);
#endif

  // token embedding, add position embedding and layer_norm
  ker_gpt_embedding_launcher<_DataType>(
      batch_size, batch_seq_len, _tw._hidden_size, _stream, _p_d_src_emb_wei[0],
      _p_d_src_emb_wei[1], _p_d_token_id, _p_d_query, _p_d_real_seq_len,
      _tw._padding_id, 0);

#ifdef DEBUG_RESULT
  print_vec(_p_d_query, "input embeddings",
            _batch_token_num * _tw._hidden_size - 5,
            _batch_token_num * _tw._hidden_size);
#endif

  for (_layer_id = 0; _layer_id < _tw._n_enc_layer; _layer_id++) {
    _weight_offset = _layer_id * _tw._weight_per_enc_layer;
    self_attention();
    ffn_add_norm();
  }

  // last layer norm
  ker_norm_layer_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_query,
      _p_d_src_emb_wei[2], _p_d_src_emb_wei[3], _max_thread_per_block);

  compute_ppl();

  return;
}

template <OperationType OpType_>
int GptEncoder<OpType_>::run_one_sample(int batch_size, int batch_seq_len) {
  _batch_size = batch_size;
  _batch_seq_len = batch_seq_len;
  _batch_token_num = batch_size * batch_seq_len;

  if (_batch_seq_len >= _tw._max_step) {
    return _batch_seq_len;
  }

  CHECK_GPU_ERROR(cudaMemcpyAsync(_p_d_real_seq_len, _h_real_seq_len.data(),
                                  sizeof(int) * _batch_size,
                                  cudaMemcpyHostToDevice, _stream));
  CHECK_GPU_ERROR(cudaMemcpyAsync(_p_d_ppl, _h_ppl.data(),
                                  sizeof(float) * _batch_size,
                                  cudaMemcpyHostToDevice, _stream));
  CHECK_GPU_ERROR(cudaMemcpyAsync(_p_d_sample_id, _p_d_token_id,
                                  sizeof(int) * _batch_size * _tw._max_step,
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
  ker_gpt_embedding_launcher<_DataType>(
      _batch_size, _batch_seq_len, _tw._hidden_size, _stream,
      _p_d_src_emb_wei[0], _p_d_src_emb_wei[1], _p_d_sample_id, _p_d_query,
      _p_d_real_seq_len, _tw._padding_id, 0);

#ifdef DEBUG_RESULT
  print_vec(_p_d_query, "embedding", _batch_token_num * _tw._hidden_size - 10,
            _batch_token_num * _tw._hidden_size);
#endif

  for (_layer_id = 0; _layer_id < _tw._n_enc_layer; _layer_id++) {
    _weight_offset = _layer_id * _tw._weight_per_enc_layer;
    self_attention(true);
    ffn_add_norm();
  }

  // last layer norm
  ker_norm_layer_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_query,
      _p_d_src_emb_wei[2], _p_d_src_emb_wei[3], _max_thread_per_block);
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
    ker_gpt_embedding_launcher<_DataType>(
        _batch_size, 1, _tw._hidden_size, _stream, _p_d_src_emb_wei[0],
        _p_d_src_emb_wei[1], _p_d_last_sample_id, _p_d_query, _p_d_real_seq_len,
        _tw._padding_id, _batch_seq_len - 1);
#ifdef DEBUG_RESULT
    print_vec(_p_d_query, "embedding", _batch_size * _tw._hidden_size - 10,
              _batch_size * _tw._hidden_size);
#endif
    for (_layer_id = 0; _layer_id < _tw._n_enc_layer; _layer_id++) {
      _weight_offset = _layer_id * _tw._weight_per_enc_layer;
      self_attention_with_cache();
      ffn_add_norm_with_cache();
    }

    // last layer norm
    ker_norm_layer_launcher<_DataType>(
        _batch_size, _tw._hidden_size, _stream, _p_d_query, _p_d_src_emb_wei[2],
        _p_d_src_emb_wei[3], _max_thread_per_block);
#ifdef DEBUG_RESULT

    print_vec(_p_d_query, "_p_d_query before logits",
              _batch_size * _tw._hidden_size - 10,
              _batch_size * _tw._hidden_size);
    if (sample_one_token_with_cache() == 0 || _batch_seq_len >= _tw._max_step)
      break;
#else
    if (sample_one_token_with_cache() == 0 || _batch_seq_len >= _tw._max_step)
      break;
#endif
  }

  CHECK_GPU_ERROR(cudaMemcpyAsync(_p_d_sample_id_buf, _p_d_sample_id,
                                  _batch_token_num * sizeof(int),
                                  cudaMemcpyDeviceToDevice, _stream));
  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));

  return _batch_seq_len;
}

template <OperationType OpType_>
int GptEncoder<OpType_>::sample_one_token() {
  /* ---step 1. project hidden states to vocab logits--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_T, CUBLAS_OP_N, _tw._src_vocab_size, _batch_token_num,
      _tw._hidden_size, &_fone, _p_d_src_emb_wei[0], _AType, _tw._hidden_size,
      _p_d_query, _BType, _tw._hidden_size, &_fzero, _p_d_logit, _CType,
      _tw._src_vocab_size, _computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#ifdef DEBUG_RESULT
  print_vec(_p_d_logit, "logits", _batch_token_num * _tw._src_vocab_size - 10,
            _batch_token_num * _tw._src_vocab_size);
#endif
  CHECK_GPU_ERROR(cudaMemsetAsync(_p_d_unfinished, 0, sizeof(int), _stream));
  /* ---step 2. sample new tokens from logits */
  if (_tw._sampling_method == "topk") {
#ifdef DEBUG_RESULT
    std::cout << "sampling using topk\n";
#endif
    ker_topk_sample_launcher<_DataType>(
        _batch_size, _batch_seq_len, _batch_seq_len, _max_thread_per_block,
        _stream, _p_d_logit, _p_d_sample_id, _p_d_sample_id_buf,
        _p_d_real_seq_len, _tw._src_vocab_size, _tw._topk, _p_d_unfinished,
        _p_d_curandstate, _tw._eos_id);
  } else {
#ifdef DEBUG_RESULT
    std::cout << "sampling using topp\n";
#endif
    ker_topp_sample_launcher<_DataType>(
        _batch_size, _batch_seq_len, _batch_seq_len, _max_thread_per_block,
        _stream, _p_d_logit, _p_d_sample_id, _p_d_sample_id_buf,
        _p_d_real_seq_len, _tw._src_vocab_size, _tw._topp, _p_d_unfinished,
        _p_d_curandstate, _tw._eos_id);
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
int GptEncoder<OpType_>::sample_one_token_with_cache() {
  /* ---step 1. project hidden states to vocab logits--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_T, CUBLAS_OP_N, _tw._src_vocab_size, _batch_size,
      _tw._hidden_size, &_fone, _p_d_src_emb_wei[0], _AType, _tw._hidden_size,
      _p_d_query, _BType, _tw._hidden_size, &_fzero, _p_d_logit, _CType,
      _tw._src_vocab_size, _computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

#ifdef DEBUG_RESULT
  print_vec(_p_d_logit, "sampling-logits",
            _batch_size * _tw._src_vocab_size - 5,
            _batch_size * _tw._src_vocab_size);
#endif

  CHECK_GPU_ERROR(cudaMemsetAsync(_p_d_unfinished, 0, sizeof(int), _stream));
  // /* ---step 2. sample new tokens from logits */
  if (_tw._sampling_method == "topk") {
#ifdef DEBUG_RESULT
    std::cout << "sampling using topk\n";
#endif
    ker_topk_sample_launcher<_DataType>(
        _batch_size, _batch_seq_len, 1, _max_thread_per_block, _stream,
        _p_d_logit, _p_d_sample_id, _p_d_sample_id_buf, _p_d_real_seq_len,
        _tw._src_vocab_size, _tw._topk, _p_d_unfinished, _p_d_curandstate,
        _tw._eos_id);
  } else {
#ifdef DEBUG_RESULT
    std::cout << "sampling using topp\n";
#endif
    ker_topp_sample_launcher<_DataType>(
        _batch_size, _batch_seq_len, 1, _max_thread_per_block, _stream,
        _p_d_logit, _p_d_sample_id, _p_d_sample_id_buf, _p_d_real_seq_len,
        _tw._src_vocab_size, _tw._topp, _p_d_unfinished, _p_d_curandstate,
        _tw._eos_id);
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
void GptEncoder<OpType_>::self_attention(bool cache) {
  /* ---step 0. layer_norm, add output_bias to "query"--- */
  ker_norm_layer_resual_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_query, _p_d_q,
      _p_d_enc_wei[_weight_offset], _p_d_enc_wei[_weight_offset + 1],
      _p_d_enc_wei[_weight_offset + 5], _max_thread_per_block);

#ifdef DEBUG_RESULT
  if (_layer_id == 0) {
    print_vec(_p_d_query, "input with bias",
              _batch_token_num * _tw._hidden_size - 5,
              _batch_token_num * _tw._hidden_size);
    print_vec(_p_d_q, "first ln output",
              _batch_token_num * _tw._hidden_size - 5,
              _batch_token_num * _tw._hidden_size);
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

#ifdef DEBUG_RESULT
  if (_layer_id == 0) {
    std::cout << "hidden_size: " << _tw._hidden_size << std::endl;
    std::cout << "_batch_token_num: " << _batch_token_num << std::endl;
    std::cout << "_dim_per_head: " << _tw._dim_per_head << std::endl;
    std::cout << "_head_num: " << _tw._head_num << std::endl;

    print_vec(_p_d_enc_wei[_weight_offset + 2], "qkv_weight_mat",
              _tw._hidden_size * _tw._hidden_size * 3 - 5,
              _tw._hidden_size * _tw._hidden_size * 3);
    print_vec(_p_d_qkv_projected, "_p_d_qkv_projected",
              _batch_token_num * _tw._hidden_size * 3 - 5,
              _batch_token_num * _tw._hidden_size * 3);
  }
#endif
  // get q, k, v by split and reshape qkv
  ker_arrange_encself_qkv_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_qkv_projected,
      _p_d_enc_wei[_weight_offset + 3], _p_d_q, _max_batch_dim, _batch_seq_len,
      _tw._dim_per_head, _tw._head_num, _max_thread_per_block);

  if (cache) {
    cudaStream_t stream;
    if (_batch_token_num > 360) {
      stream = _cache_stream;
      CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
    } else {
      stream = _stream;
    }
    CHECK_GPU_ERROR(
        cudaMemcpyAsync(_p_d_k_cache + _layer_id * _max_batch_dim, _p_d_k,
                        _batch_token_num * _tw._hidden_size * sizeof(_DataType),
                        cudaMemcpyDeviceToDevice, stream));
    CHECK_GPU_ERROR(
        cudaMemcpyAsync(_p_d_v_cache + _layer_id * _max_batch_dim, _p_d_v,
                        _batch_token_num * _tw._hidden_size * sizeof(_DataType),
                        cudaMemcpyDeviceToDevice, stream));
  }

#ifdef DEBUG_RESULT
  if (_layer_id == 0) {
    print_vec(_p_d_q, "_p_d_q", _batch_token_num * _tw._hidden_size - 5,
              _batch_token_num * _tw._hidden_size);
    print_vec(_p_d_k, "_p_d_k", _batch_token_num * _tw._hidden_size - 5,
              _batch_token_num * _tw._hidden_size);
    print_vec(_p_d_v, "_p_d_v", _batch_token_num * _tw._hidden_size - 5,
              _batch_token_num * _tw._hidden_size);
  }
#endif

  /* ---step 2. correlation = q * k, perform softmax on correlation--- */
  CHECK_GPU_ERROR(cublasGemmStridedBatchedEx(
      _hd, CUBLAS_OP_T, CUBLAS_OP_N, _batch_seq_len, _batch_seq_len,
      _tw._dim_per_head, &_atten_scaler, _p_d_k, _AType, _tw._dim_per_head,
      _batch_seq_len * _tw._dim_per_head, _p_d_q, _BType, _tw._dim_per_head,
      _batch_seq_len * _tw._dim_per_head, &_fzero, _p_d_c, _CType,
      _batch_seq_len, _batch_seq_len * _batch_seq_len,
      _batch_size * _tw._head_num, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

#ifdef DEBUG_RESULT
  if (_layer_id == 0) {
    print_vec(_p_d_c, "q*k",
              _batch_token_num * _batch_seq_len * _tw._head_num - 5,
              _batch_token_num * _batch_seq_len * _tw._head_num);
  }
#endif

  ker_correlation_softmax_gpt_launcher<_DataType>(_batch_size, _batch_seq_len,
                                                  _tw._head_num, _stream,
                                                  _p_d_c, _p_d_real_seq_len);

#ifdef DEBUG_RESULT
  if (_layer_id == 0) {
    print_vec(_p_d_c, "mask weights",
              _batch_token_num * _batch_seq_len * _tw._head_num - 5,
              _batch_token_num * _batch_seq_len * _tw._head_num);
  }
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

#ifdef DEBUG_RESULT
  if (_layer_id == 0) {
    print_vec(_p_d_q, "value after attention",
              _batch_token_num * _tw._hidden_size - 5,
              _batch_token_num * _tw._hidden_size);
  }
#endif

  // use v to save reshaped q, since they are in same size and v
  // will not be use again before the next multi-head-attention
  ker_arrange_atten_output_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_q, _p_d_v,
      _batch_seq_len, _tw._dim_per_head, _tw._head_num, _max_thread_per_block);

#ifdef DEBUG_RESULT
  if (_layer_id == 0) {
    print_vec(_p_d_v, "reshaped value after attention", 0, 5);
    print_vec(_p_d_query, "attention input with output bias", 0, 5);
  }
#endif

  /* ---step 4. new_q = ori_q + new_q * output_wei--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size, _batch_token_num,
      _tw._hidden_size, &_fone, _p_d_enc_wei[_weight_offset + 4], _AType,
      _tw._hidden_size, _p_d_v, _BType, _tw._hidden_size, &_fone, _p_d_query,
      _CType, _tw._hidden_size, _computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

#ifdef DEBUG_RESULT
  if (_layer_id == 0) {
    print_vec(_p_d_enc_wei[_weight_offset + 4], "attn out kernel", 0, 5);
    print_vec(_p_d_query, "attention output", 0, 5);
  }
#endif
  return;
}

template <OperationType OpType_>
void GptEncoder<OpType_>::self_attention_with_cache() {
  _DataType *_p_d_k_cache_cur_layer = _p_d_k_cache + _layer_id * _max_batch_dim;
  _DataType *_p_d_v_cache_cur_layer = _p_d_v_cache + _layer_id * _max_batch_dim;

#ifdef DEBUG_RESULT
  if (_layer_id == 0) {
    print_vec(_p_d_k_cache_cur_layer, "_p_d_k_cache_cur_layer",
              _batch_size * (_batch_seq_len - 1) * _tw._hidden_size - 5,
              _batch_size * (_batch_seq_len - 1) * _tw._hidden_size);
    print_vec(_p_d_v_cache_cur_layer, "_p_d_v_cache_cur_layer",
              _batch_size * (_batch_seq_len - 1) * _tw._hidden_size - 5,
              _batch_size * (_batch_seq_len - 1) * _tw._hidden_size);
  }
#endif

  /* ---step 0. layer_norm, add output_bias to "query"--- */
  ker_norm_layer_resual_launcher<_DataType>(
      _batch_size, _tw._hidden_size, _stream, _p_d_query, _p_d_q,
      _p_d_enc_wei[_weight_offset], _p_d_enc_wei[_weight_offset + 1],
      _p_d_enc_wei[_weight_offset + 5], _max_thread_per_block);

#ifdef DEBUG_RESULT
  if (_layer_id == 0) {
    print_vec(_p_d_query, "input with bias", _batch_size * _tw._hidden_size - 5,
              _batch_size * _tw._hidden_size);
    print_vec(_p_d_q, "first ln output", _batch_size * _tw._hidden_size - 5,
              _batch_size * _tw._hidden_size);
  }
#endif

  /* ---step 1. qkv = ori_q * qkv_wei + bias, and reshape qkv for multi-head
   * gemm--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size * 3, _batch_size,
      _tw._hidden_size, &_fone, _p_d_enc_wei[_weight_offset + 2], _AType,
      _tw._hidden_size * 3, _p_d_q, _BType, _tw._hidden_size, &_fzero,
      _p_d_qkv_projected, _CType, _tw._hidden_size * 3, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

#ifdef DEBUG_RESULT
  if (_layer_id == 0) {
    print_vec(_p_d_qkv_projected, "_p_d_qkv_projected",
              _batch_size * _tw._hidden_size * 3 - 5,
              _batch_size * _tw._hidden_size * 3);
  }
#endif
  // get q, k, v by split and reshape qkv
  ker_arrange_qkv_with_cache_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_qkv_projected,
      _p_d_enc_wei[_weight_offset + 3], _p_d_q, _p_d_k, _p_d_k_cache_cur_layer,
      _p_d_v, _p_d_v_cache_cur_layer, _max_batch_dim, _batch_seq_len,
      _tw._dim_per_head, _tw._head_num);

  // copy new k and v to cache
  cudaStream_t stream;
  if (_batch_token_num > 360) {
    stream = _cache_stream;
    CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  } else {
    stream = _stream;
  }
  CHECK_GPU_ERROR(
      cudaMemcpyAsync(_p_d_k_cache_cur_layer, _p_d_k,
                      _batch_token_num * _tw._hidden_size * sizeof(_DataType),
                      cudaMemcpyDeviceToDevice, stream));
  CHECK_GPU_ERROR(
      cudaMemcpyAsync(_p_d_v_cache_cur_layer, _p_d_v,
                      _batch_token_num * _tw._hidden_size * sizeof(_DataType),
                      cudaMemcpyDeviceToDevice, stream));
#ifdef DEBUG_RESULT
  if (_layer_id == 0) {
    print_vec(_p_d_q, "_p_d_q", _batch_size * _tw._hidden_size - 5,
              _batch_size * _tw._hidden_size);
    print_vec(_p_d_k, "_p_d_k", _batch_token_num * _tw._hidden_size - 5,
              _batch_token_num * _tw._hidden_size);
    print_vec(_p_d_v, "_p_d_v", _batch_token_num * _tw._hidden_size - 5,
              _batch_token_num * _tw._hidden_size);
  }
#endif

  /* ---step 2. correlation = q * k, perform softmax on correlation
  correlation: [batch_size, heads_num, 1, batch_seq_len]--- */
  CHECK_GPU_ERROR(cublasGemmStridedBatchedEx(
      _hd, CUBLAS_OP_T, CUBLAS_OP_N, _batch_seq_len, 1, _tw._dim_per_head,
      &_atten_scaler, _p_d_k, _AType, _tw._dim_per_head,
      _batch_seq_len * _tw._dim_per_head, _p_d_q, _BType, _tw._dim_per_head,
      _tw._dim_per_head, &_fzero, _p_d_c, _CType, _batch_seq_len,
      _batch_seq_len, _batch_size * _tw._head_num, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

#ifdef DEBUG_RESULT
  if (_layer_id == 0) {
    print_vec(_p_d_c, "q*k", _batch_size * _batch_seq_len * _tw._head_num - 5,
              _batch_size * _batch_seq_len * _tw._head_num);
  }
#endif
  ker_attention_mask_weights_launcher<_DataType>(_batch_size, 1, _batch_seq_len,
                                                 _tw._head_num, _stream, _p_d_c,
                                                 _p_d_real_seq_len);

#ifdef DEBUG_RESULT
  if (_layer_id == 0) {
    print_vec(_p_d_c, "mask weights",
              _batch_size * _batch_seq_len * _tw._head_num - 5,
              _batch_size * _batch_seq_len * _tw._head_num);
  }
#endif

  /* ---step 3. new_q = correlation * v--- */
  CHECK_GPU_ERROR(cublasGemmStridedBatchedEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._dim_per_head, 1, _batch_seq_len,
      &_fone, _p_d_v, _AType, _tw._dim_per_head,
      _batch_seq_len * _tw._dim_per_head, _p_d_c, _BType, _batch_seq_len,
      _batch_seq_len, &_fzero, _p_d_q, _CType, _tw._dim_per_head,
      _tw._dim_per_head, _batch_size * _tw._head_num, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

#ifdef DEBUG_RESULT
  if (_layer_id == 0) {
    print_vec(_p_d_q, "value after attention",
              _batch_size * _tw._hidden_size - 5,
              _batch_size * _tw._hidden_size);
  }
#endif
  // use v to save reshaped q, since they are in same size and v
  // will not be use again before the next multi-head-attention
  ker_arrange_atten_output_launcher<_DataType>(
      _batch_size, _tw._hidden_size, _stream, _p_d_q, _p_d_v, 1,
      _tw._dim_per_head, _tw._head_num, _max_thread_per_block);

#ifdef DEBUG_RESULT
  if (_layer_id == 0) {
    print_vec(_p_d_v, "reshaped value after attention", 0, 5);
    print_vec(_p_d_query, "attention input with output bias", 0, 5);
  }
#endif

  /* ---step 4. new_q = ori_q + new_q * output_wei--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size, _batch_size,
      _tw._hidden_size, &_fone, _p_d_enc_wei[_weight_offset + 4], _AType,
      _tw._hidden_size, _p_d_v, _BType, _tw._hidden_size, &_fone, _p_d_query,
      _CType, _tw._hidden_size, _computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

#ifdef DEBUG_RESULT
  if (_layer_id == 0) {
    print_vec(_p_d_enc_wei[_weight_offset + 4], "attn out kernel", 0, 5);
    print_vec(_p_d_query, "attention output", 0, 5);
  }
#endif
  return;
}

template <OperationType OpType_>
void GptEncoder<OpType_>::ffn_add_norm() {
  /* ---step 0. layer_norm, add output_bias to "query"--- */
  ker_norm_layer_resual_launcher<_DataType>(
      _batch_token_num, _tw._hidden_size, _stream, _p_d_query, _p_d_ffn_buf1,
      _p_d_enc_wei[_weight_offset + 6], _p_d_enc_wei[_weight_offset + 7],
      _p_d_enc_wei[_weight_offset + 11], _max_thread_per_block);

  /* ---step 1. first ffn layer--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._inner_size, _batch_token_num,
      _tw._hidden_size, &_fone, _p_d_enc_wei[_weight_offset + 8], _AType,
      _tw._inner_size, _p_d_ffn_buf1, _BType, _tw._hidden_size, &_fzero,
      _p_d_ffn_buf2, _CType, _tw._inner_size, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  ker_bias_gelu_launcher<_DataType>(
      _batch_token_num, _max_thread_per_block, _stream, _p_d_ffn_buf2,
      _p_d_enc_wei[_weight_offset + 9], _tw._inner_size);

  /* ---step 2. second ffn layer--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size, _batch_token_num,
      _tw._inner_size, &_fone, _p_d_enc_wei[_weight_offset + 10], _AType,
      _tw._hidden_size, _p_d_ffn_buf2, _BType, _tw._inner_size, &_fone,
      _p_d_query, _CType, _tw._hidden_size, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  return;
}

template <OperationType OpType_>
void GptEncoder<OpType_>::ffn_add_norm_with_cache() {
  /* ---step 0. layer_norm, add output_bias to "query"--- */
  ker_norm_layer_resual_launcher<_DataType>(
      _batch_size, _tw._hidden_size, _stream, _p_d_query, _p_d_ffn_buf1,
      _p_d_enc_wei[_weight_offset + 6], _p_d_enc_wei[_weight_offset + 7],
      _p_d_enc_wei[_weight_offset + 11], _max_thread_per_block);

  /* ---step 1. first ffn layer--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._inner_size, _batch_size,
      _tw._hidden_size, &_fone, _p_d_enc_wei[_weight_offset + 8], _AType,
      _tw._inner_size, _p_d_ffn_buf1, _BType, _tw._hidden_size, &_fzero,
      _p_d_ffn_buf2, _CType, _tw._inner_size, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  ker_bias_gelu_launcher<_DataType>(
      _batch_size, _max_thread_per_block, _stream, _p_d_ffn_buf2,
      _p_d_enc_wei[_weight_offset + 9], _tw._inner_size);

  /* ---step 2. second ffn layer--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size, _batch_size,
      _tw._inner_size, &_fone, _p_d_enc_wei[_weight_offset + 10], _AType,
      _tw._hidden_size, _p_d_ffn_buf2, _BType, _tw._inner_size, &_fone,
      _p_d_query, _CType, _tw._hidden_size, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  return;
}

/**
Compute ppl from encoder output
*/
template <OperationType OpType_>
void GptEncoder<OpType_>::compute_ppl() {
  /* ---step 1. project hidden states to vocab logits--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_T, CUBLAS_OP_N, _tw._src_vocab_size, _batch_token_num,
      _tw._hidden_size, &_fone, _p_d_src_emb_wei[0], _AType, _tw._hidden_size,
      _p_d_query, _BType, _tw._hidden_size, &_fzero, _p_d_logit, _CType,
      _tw._src_vocab_size, _computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

#ifdef DEBUG_RESULT
  print_vec(_p_d_logit, "logits", _batch_token_num * _tw._src_vocab_size - 5,
            _batch_token_num * _tw._src_vocab_size);
#endif

  /* ---step 2. compute language model ppl--- */
  ker_ppl_launcher<_DataType>(
      _batch_size, _batch_seq_len, _max_thread_per_block, _stream, _p_d_logit,
      _p_d_token_id, _p_d_real_seq_len, _p_d_ppl, _tw._src_vocab_size);
}

template class GptEncoder<OperationType::FP16>;
template class GptEncoder<OperationType::FP32>;

}  // namespace cuda
}  // namespace lightseq
