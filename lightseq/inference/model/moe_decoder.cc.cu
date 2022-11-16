#include "moe_decoder.h"
#include "../kernels/transformerKernels.h"
#include "../kernels/embKernels.h"
#include "../kernels/moeKernels.h"

/**
@file
Moe decoder, composed by gemm lib and
  custom cuda kernel function
*/

namespace lightseq {
namespace cuda {

template <OperationType OpType_>
MoeDecoder<OpType_>::MoeDecoder(int max_batch_size, const int* p_d_padding_mask,
                                const _DataType* p_d_encoder_output,
                                int* p_d_result, MoeWeight<OpType_>& tw,
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
      _atten_scaler(sqrt(1.f / tw._dim_per_head)),
      _logit_scaler(_tw._no_scale_embedding ? 1.f
                                            : sqrt(1.f / tw._hidden_size)),
      _h_alive_seq_probs(max_batch_size * tw._beam_size,
                         min_log_probability / 2),
      _h_length_norm(tw._max_step, 1.f),
      _h_unfinished(1),
      _gate_weight_offset(0),
      _p_d_dec_gate_wei(tw.get_dec_gate_wei()),
      _max_step_token_num(max_batch_size * tw._beam_size) {
  for (int i = 0; i < _h_alive_seq_probs.size(); i += tw._beam_size) {
    _h_alive_seq_probs[i] = 0.f;
  }
  if (tw._length_penalty >= 0) {
    for (int i = 0; i < _h_length_norm.size(); i++) {
      _h_length_norm[i] = length_norm(i + 1, tw._length_penalty);
    }
  }
  return;
}

/**
Compute GPU memory size needed by moe decoder,
  to see how these memory is used, checkout init_buffer() for detail
*/
template <OperationType OpType_>
long MoeDecoder<OpType_>::compute_buffer_bytesize() {
  long cache_bytesize = 4 * _tw._n_dec_layer * _layer_size_self_k +
                        2 * _tw._n_dec_layer * _layer_size_encdec_k +
                        _max_batch_size * _tw._beam_size * _tw._hidden_size;
  cache_bytesize *= sizeof(_DataType);

  long decode_buffer_bytesize =
      _max_batch_size * _tw._beam_size * _tw._hidden_size * 4 +
      _max_batch_size * _tw._beam_size *
          max(_tw._hidden_size, _tw._inner_size) +
      _max_batch_size * _tw._head_num * _tw._beam_size * _tw._max_step +
      _max_step_token_num * _tw._expert_num_decoder *
          (_tw._hidden_size + _tw._inner_size + 1);
  decode_buffer_bytesize *= sizeof(_DataType);
  decode_buffer_bytesize +=
      (_max_step_token_num * _tw._expert_num_decoder * sizeof(float) +
       _tw._moe_topk_decoder * _max_step_token_num * sizeof(int));

  long sf = _max_batch_size * _tw._beam_size * _tw._trg_vocab_size * 2 +
            _max_batch_size * _tw._beam_size * 2;
  long si = _max_batch_size * _tw._beam_size * _tw._max_step * 2 +
            _max_batch_size * _tw._beam_size * _tw._trg_vocab_size +
            _max_batch_size * _tw._beam_size + 1;
  long beam_buffer_bytesize = sf * sizeof(float) + si * sizeof(int);

  return cache_bytesize + max(decode_buffer_bytesize, beam_buffer_bytesize);
}

/**
Init the GPU memory pointer which point to
  the memory buffer needed by decoder.
These buffer are used during custom cuda kernel function,
  find the corresponding function to see how these buffer are used
*/
template <OperationType OpType_>
void MoeDecoder<OpType_>::init_buffer(void* pbuf) {
  std::cout << "decoder buffer init start" << std::endl;
  _DataType* curp = reinterpret_cast<_DataType*>(pbuf);

  for (int i = 0; i < _tw._n_dec_layer; i++) {
    // encoder ouput after project, the "key" of enc_dec attention
    _p_d_encdec_k_bgeem.push_back(curp);
    curp += _layer_size_encdec_k;
  }
  for (int i = 0; i < _tw._n_dec_layer; i++) {
    // encoder ouput after project, the "value" of enc_dec attention
    _p_d_encdec_v_bgeem.push_back(curp);
    curp += _layer_size_encdec_k;
  }
  // reused buffer with _p_d_self_k_bgeem _p_d_self_v_bgeem,
  // no need to add curp because _p_d_encoder_out_buf is smaller
  // and no need to use it any more after get _p_d_encdec_k_bgeem
  // and _p_d_encdec_v_bgeem
  _p_d_encoder_out_buf = curp;

  for (int i = 0; i < _tw._n_dec_layer * 2; i++) {
    // the "key" of decoder self attention, we need to maintain it by twice
    // one for current step's "key", one for "key" of beam_search cache
    // after finishing current step's search, we will copy the first one
    // to the second one to refresh beam_search cache
    // based on the selected beam id
    _p_d_self_k_bgeem.push_back(curp);
    curp += _layer_size_self_k;
  }
  for (int i = 0; i < _tw._n_dec_layer * 2; i++) {
    // the "value" of decoder self attention, we need to maintain it by twice
    // one for current step's "value", one for "value" of beam_search cache
    // after finishing current step's search, we will copy the first one
    // to the second one to refresh beam_search cache
    // based on the selected beam id
    _p_d_self_v_bgeem.push_back(curp);
    curp += _layer_size_self_k;
  }
  _p_d_self_k_bgeem1 = _p_d_self_k_bgeem.data();
  _p_d_self_k_bgeem2 = _p_d_self_k_bgeem.data() + _tw._n_dec_layer;
  _p_d_self_v_bgeem1 = _p_d_self_v_bgeem.data();
  _p_d_self_v_bgeem2 = _p_d_self_v_bgeem.data() + _tw._n_dec_layer;

  // GPU memory buffer to save "query",
  // In all layers, using the same buffer
  _p_d_cur_step_query = curp;
  curp += _max_batch_size * _tw._beam_size * _tw._hidden_size;

  // we can use the same buffer for decoder network computation
  // and beam search, since they're serial.
  _DataType* reuse_p = curp;

  // for decode network computation
  _p_d_self_step_qkv = curp;  // [q, k, v], result of gemm
  curp += _max_batch_size * _tw._beam_size * _tw._hidden_size * 3;
  _p_d_query_buf1 = curp;  // "query" buffer
  curp += _max_batch_size * _tw._beam_size * _tw._hidden_size;
  _p_d_query_buf2 = curp;  // "query" buffer
  curp +=
      _max_batch_size * _tw._beam_size * max(_tw._hidden_size, _tw._inner_size);
  _p_d_c = curp;  // correlation(attention score) buffer
  curp += _max_batch_size * _tw._head_num * _tw._beam_size * _tw._max_step;
  _p_d_gate = curp;  // moe gate buffer
  curp += _tw._expert_num_decoder * _max_step_token_num;
  _p_d_moe_input_buf = curp;  // moe input buffer
  curp += _tw._expert_num_decoder * _max_step_token_num * _tw._hidden_size;
  _p_d_moe_inner_buf = curp;  // moe ffns buffer
  curp += _tw._expert_num_decoder * _max_step_token_num * _tw._inner_size;
  _p_d_score_routed = reinterpret_cast<float*>(curp);  // expert routing score
  // ids of routed experts in moe
  _p_d_expert_id_routed = reinterpret_cast<int*>(
      _p_d_score_routed + _max_step_token_num * _tw._expert_num_decoder);

  // for beam search
  curp = reuse_p;
  _p_d_logit_buf = curp;  // vocab ligit
  curp += _max_batch_size * _tw._beam_size * _tw._trg_vocab_size;
  // always be float
  float* fcurp = (float*)curp;
  // seq score ended with every target token for current step
  _p_d_can_score = fcurp;
  fcurp += _max_batch_size * _tw._beam_size * _tw._trg_vocab_size;
  _p_d_alive_seq_probs = fcurp;  // alive seq probability
  fcurp += _max_batch_size * _tw._beam_size;
  _p_d_alive_seq_score = fcurp;  // alive seq score
  fcurp += _max_batch_size * _tw._beam_size;

  int* pint = reinterpret_cast<int*>(fcurp);
  // FIXME
  std::vector<int> start_id_vec(
      _max_batch_size * _tw._beam_size * _tw._max_step * 2, _tw._start_id);
  usleep(3000);
  CHECK_GPU_ERROR(cudaMemcpyAsync(pint, start_id_vec.data(),
                                  sizeof(int) * start_id_vec.size(),
                                  cudaMemcpyHostToDevice, _stream));
  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  // token id of alive seq, we need to maintain it by twice
  // one for current step, one for beam_search cache
  // after finishing current step's search, we will copy the first one
  // to the second one to refresh beam_search cache
  // based on the selected beam id
  _p_d_alive_seq = pint;
  pint += _max_batch_size * _tw._beam_size * _tw._max_step;
  _p_d_alive_seq_buf = pint;
  pint += _max_batch_size * _tw._beam_size * _tw._max_step;

  // candidate token id for every beam, selected by rough top-beam_size op
  _p_d_can_idx = pint;
  pint += _max_batch_size * _tw._beam_size * _tw._trg_vocab_size;
  // candidate token number for every beam, selected by rough top-beam_size op
  _p_d_can_num = pint;
  pint += _max_batch_size * _tw._beam_size + 1;

  CHECK_GPU_ERROR(cudaMalloc((void**)&_p_d_sample_unfinished, sizeof(int)));
  CHECK_GPU_ERROR(cudaMalloc((void**)&_p_d_curandstate,
                             _max_batch_size * sizeof(curandState)));
  ker_curand_setup<<<_max_batch_size, 1, 0, _stream>>>(_p_d_curandstate);
  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  CHECK_GPU_ERROR(cudaGetLastError());
  std::cout << "decoder buffer init succeed" << std::endl;
  return;
}

/**
Some requirements needed by custom cuda kernel function
*/
template <OperationType OpType_>
std::string MoeDecoder<OpType_>::check() {
  // if (_max_thread_per_block < _tw._hidden_size) {
  //   return "violate hidden_size <= max_thread_per_block";
  // }
  if (_tw._inner_size & 1) {
    return "violate inner_size % 2 = 0";
  }
  if (_tw._dim_per_head & 1) {
    std::cout << "decoder_dim_per_head:" << _tw._dim_per_head << std::endl;
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
  if (_tw._moe_topk_decoder != 1 && _tw._moe_topk_decoder != 2) {
    return "moe topk should be 1 or 2";
  }
  if (_tw._expert_num_decoder > 1024) {
    return "number of moe expert should not be greater than 1024";
  }
  return "";
}

/**
Decoder inference
*/
template <OperationType OpType_>
void MoeDecoder<OpType_>::run_one_infer(int batch_size, int batch_seq_len) {
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
void MoeDecoder<OpType_>::project_encoder_output() {
  int kv_dim = _tw._hidden_size * 2 * _tw._n_dec_layer;
#ifdef DEBUG_RESULT
  CHECK_GPU_ERROR(cudaStreamSynchronize(_stream));
  print_vec(_p_d_encoder_output, "_p_d_encoder_output(head):", 5);
  print_vec(_p_d_encoder_output + _batch_token_num * _tw._hidden_size - 5,
            "_p_d_encoder_output(tail)", 5);
  print_vec(_p_d_trg_emb_wei[4], "encoder project(head):", 10);
#endif
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, kv_dim, _batch_token_num, _tw._hidden_size,
      &_type_one, _p_d_trg_emb_wei[4], _AType, kv_dim, _p_d_encoder_output,
      _BType, _tw._hidden_size, &_type_zero, _p_d_encoder_out_buf, _CType,
      kv_dim, _computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
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
      _p_d_encoder_out_buf, _p_d_trg_emb_wei[5], _p_d_encdec_k_bgeem[0],
      _p_d_encdec_v_bgeem[0], _layer_size_encdec_k, _batch_seq_len,
      _tw._dim_per_head, _tw._head_num, _max_thread_per_block);
  return;
}

/**
Decode one step
*/
template <OperationType OpType_>
bool MoeDecoder<OpType_>::run_step() {
  embedding();
  decoder_stack();
  /* --- Project hidden states to vocab logits--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._trg_vocab_size, _step_token_num,
      _tw._hidden_size, &_logit_scaler, _p_d_trg_emb_wei[0], _AType,
      _tw._trg_vocab_size, _p_d_cur_step_query, _BType, _tw._hidden_size,
      //&_type_zero, _p_d_logit_buf, _CType, _tw._trg_vocab_size, _computeType,
      &_fzero, _p_d_logit_buf, _CType, _tw._trg_vocab_size, CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#ifdef DEBUG_RESULT
  for (int i = 0; i < _batch_size; i++) {       // batch_id
    for (int j = 0; j < _tw._beam_size; j++) {  // beam_id
      std::cout << "decoder output: batch-" << i << ", beam-" << j << std::endl;
      print_vec(_p_d_cur_step_query + i * _tw._beam_size * _tw._hidden_size +
                    j * _tw._hidden_size,
                "hidden", 10);
      print_vec(_p_d_logit_buf + i * _tw._beam_size * _tw._trg_vocab_size +
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
void MoeDecoder<OpType_>::embedding() {
  // _p_d_trg_emb_wei: {token_emb, position_emb, norm_scale, norm_bias,
  // enc_out_kernel_kv, enc_out_bias_kv, logit_bias}
  launch_dec_emb<_DataType>(_p_d_trg_emb_wei[0], _p_d_trg_emb_wei[1],
                            _p_d_alive_seq, _p_d_trg_emb_wei[7], _p_d_lang_id,
                            _p_d_cur_step_query, _batch_size, _tw._beam_size,
                            _tw._hidden_size, _tw._trg_vocab_size, _cur_step,
                            _tw._max_step, _tw._multilg_type, _stream);
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
Decoder feedforward, composed by self_atten,
  enc-dec-atten, ffn
*/
template <OperationType OpType_>
void MoeDecoder<OpType_>::decoder_stack() {
  // _p_d_dec_wei = {self_norm_scale, self_norm_bias,
  // self_qkv_kernel, self_qkv_bias, self_output_kernel, self_output_bias
  // encdec_norm_scale, encdec_norm_bias,
  // encdec_q_kernel, encdec_q_bias, encdec_output_kernel, encdec_output_bias
  // ffn_norm_scale, ffn_norm_bias, ffn_first_kernel, ffn_first_bias,
  // ffn_second_kernel, ffn_second_bias} * encoder_layer_num
  _gate_weight_offset = 0;
  for (_layer_id = 0; _layer_id < _tw._n_dec_layer; _layer_id++) {
    _weight_offset = _layer_id * _tw._weight_per_dec_layer;

    self_attention();

    encdec_attention();

    ffn_add_norm();
  }

  // last layer norm
  ker_norm_layer_launcher<_DataType>(
      _step_token_num, _tw._hidden_size, _stream, _p_d_cur_step_query,
      _p_d_trg_emb_wei[2], _p_d_trg_emb_wei[3], _max_thread_per_block);
  return;
}

/**
Decoder self attention
*/
template <OperationType OpType_>
void MoeDecoder<OpType_>::self_attention() {
  /* ---step 0. layer_norm, add output_bias to "query"--- */
  ker_norm_layer_resual_launcher<_DataType>(
      _step_token_num, _tw._hidden_size, _stream, _p_d_cur_step_query,
      _p_d_query_buf1, _p_d_dec_wei[_weight_offset],
      _p_d_dec_wei[_weight_offset + 1], _p_d_dec_wei[_weight_offset + 5],
      _max_thread_per_block, _tw._is_post_ln);

#ifdef DEBUG_RESULT
  print_vec(_p_d_query_buf1, "self attn ln(head): ", 5);
  print_vec(_p_d_query_buf1 + _step_token_num * _tw._hidden_size - 5,
            "self attn ln(tail): ", 5);
#endif

  /* ---step 1. qkv = ori_q * qkv_wei + bias, and reshape qkv for multi-head
   * gemm--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size * 3, _step_token_num,
      _tw._hidden_size, &_type_one, _p_d_dec_wei[_weight_offset + 2], _AType,
      _tw._hidden_size * 3, _p_d_query_buf1, _BType, _tw._hidden_size,
      &_type_zero, _p_d_self_step_qkv, _CType, _tw._hidden_size * 3,
      _computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

#ifdef DEBUG_RESULT
  print_vec(_p_d_self_step_qkv, "self qkv(head): ", 5);
  print_vec(_p_d_self_step_qkv + _step_token_num * _tw._hidden_size * 3 - 5,
            "self qkv(tail): ", 5);
#endif

  // get q, k, v by split and reshape qkv
  ker_arrange_decself_qkv_launcher<_DataType>(
      _step_token_num, _tw._hidden_size, _stream, _p_d_self_step_qkv,
      _p_d_dec_wei[_weight_offset + 3], _p_d_query_buf1,
      _p_d_self_k_bgeem1[_layer_id], _p_d_self_v_bgeem1[_layer_id],
      _tw._head_num, _tw._dim_per_head, _tw._max_step, _cur_step,
      _max_thread_per_block);

#ifdef DEBUG_RESULT
  print_vec(_p_d_query_buf1, "self attn q(head): ", 5);
  print_vec(_p_d_query_buf1 + _step_token_num * _tw._hidden_size - 5,
            "self attn q(tail): ", 5);
  print_vec(_p_d_self_k_bgeem1[_layer_id] +
                _cur_step * _tw._hidden_size / _tw._head_num,
            "self attn k(head): ", 5);
  print_vec(
      _p_d_self_k_bgeem1[_layer_id] +
          _step_token_num * _tw._hidden_size * _tw._max_step -
          ((_tw._max_step - _cur_step - 1) * _tw._hidden_size / _tw._head_num) -
          5,
      "self attn k(tail): ", 5);
  print_vec(_p_d_self_v_bgeem1[_layer_id] +
                _cur_step * _tw._hidden_size / _tw._head_num,
            "self attn v(head): ", 5);
  print_vec(
      _p_d_self_v_bgeem1[_layer_id] +
          _step_token_num * _tw._hidden_size * _tw._max_step -
          ((_tw._max_step - _cur_step - 1) * _tw._hidden_size / _tw._head_num) -
          5,
      "self attn v(tail): ", 5);
#endif

  /* ---step 2. correlation = q * k, perform softmax on correlation--- */
  CHECK_GPU_ERROR(cublasGemmStridedBatchedEx(
      _hd, CUBLAS_OP_T, CUBLAS_OP_N, _cur_step + 1, 1, _tw._dim_per_head,
      &_atten_scaler, _p_d_self_k_bgeem1[_layer_id], _AType, _tw._dim_per_head,
      _tw._max_step * _tw._dim_per_head, _p_d_query_buf1, _BType,
      _tw._dim_per_head, _tw._dim_per_head, &_type_zero, _p_d_c, _CType,
      _cur_step + 1, _cur_step + 1, _step_token_num * _tw._head_num,
      _computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  ker_correlation_softmax_decself_launcher(_step_token_num * _tw._head_num,
                                           _cur_step + 1, _stream, _p_d_c);

#ifdef DEBUG_RESULT
  print_vec(_p_d_c, "self attn corr(head): ", 5);
  print_vec(_p_d_c + _step_token_num * _tw._head_num * (_cur_step + 1) - 5,
            "self attn corr(tail): ", 5);
#endif

  /* ---step 3. new_q = correlation * v--- */
  CHECK_GPU_ERROR(cublasGemmStridedBatchedEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._dim_per_head, 1, _cur_step + 1,
      &_type_one, _p_d_self_v_bgeem1[_layer_id], _AType, _tw._dim_per_head,
      _tw._max_step * _tw._dim_per_head, _p_d_c, _BType, _cur_step + 1,
      _cur_step + 1, &_type_zero, _p_d_query_buf1, _CType, _tw._dim_per_head,
      _tw._dim_per_head, _step_token_num * _tw._head_num, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

#ifdef DEBUG_RESULT
  print_vec(_p_d_query_buf1, "self attn before ffn(head): ", 5);
  print_vec(_p_d_query_buf1 + _step_token_num * _tw._hidden_size - 5,
            "self attn before ffn(tail): ", 5);
#endif

  /* ---step 4. new_q = ori_q + new_q * output_wei--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size, _step_token_num,
      _tw._hidden_size, &_type_one, _p_d_dec_wei[_weight_offset + 4], _AType,
      _tw._hidden_size, _p_d_query_buf1, _BType, _tw._hidden_size, &_type_one,
      _p_d_cur_step_query, _CType, _tw._hidden_size, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

#ifdef DEBUG_RESULT
  print_vec(_p_d_cur_step_query, "self attn out(head): ", 3);
  print_vec(_p_d_cur_step_query + _step_token_num * _tw._hidden_size - 3,
            "self attn out(tail): ", 3);
#endif
}

/**
Encode-Decoder attention
*/
template <OperationType OpType_>
void MoeDecoder<OpType_>::encdec_attention() {
  /* ---step 0. layer_norm, add output_bias to "query"--- */
  ker_norm_layer_resual_launcher<_DataType>(
      _step_token_num, _tw._hidden_size, _stream, _p_d_cur_step_query,
      _p_d_query_buf1, _p_d_dec_wei[_weight_offset + 6],
      _p_d_dec_wei[_weight_offset + 7], _p_d_dec_wei[_weight_offset + 11],
      _max_thread_per_block, _tw._is_post_ln);

#ifdef DEBUG_RESULT
  print_vec(_p_d_query_buf1, "encdec attn ln(head): ", 5);
  print_vec(_p_d_query_buf1 + _step_token_num * _tw._hidden_size - 5,
            "encdec attn ln(tail): ", 5);
#endif

  /* ---step 1. new_q = ori_q * q_wei + bias, reshape new_q for multi-head
   * gemm--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size, _step_token_num,
      _tw._hidden_size, &_type_one, _p_d_dec_wei[_weight_offset + 8], _AType,
      _tw._hidden_size, _p_d_query_buf1, _BType, _tw._hidden_size, &_type_zero,
      _p_d_query_buf2, _CType, _tw._hidden_size, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  ker_arrange_encdec_q_launcher<_DataType>(
      _step_token_num, _tw._hidden_size, _stream, _p_d_query_buf2,
      _p_d_dec_wei[_weight_offset + 9], _p_d_query_buf1, _tw._beam_size,
      _tw._dim_per_head, _tw._head_num, _max_thread_per_block);

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
  ker_arrange_atten_output_launcher<_DataType>(
      _step_token_num, _tw._hidden_size, _stream, _p_d_query_buf1,
      _p_d_query_buf2, _tw._beam_size, _tw._dim_per_head, _tw._head_num,
      _max_thread_per_block);

  /* ---step 4. new_q = ori_q + new_q * output_wei--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size, _step_token_num,
      _tw._hidden_size, &_type_one, _p_d_dec_wei[_weight_offset + 10], _AType,
      _tw._hidden_size, _p_d_query_buf2, _BType, _tw._hidden_size, &_type_one,
      _p_d_cur_step_query, _CType, _tw._hidden_size, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  return;
}

template <OperationType OpType_>
void MoeDecoder<OpType_>::ffn_add_norm() {
  if (_tw._is_moe_layer_decoder[_layer_id]) {
    /*
     * _gate_type
     * 0:soft gate
     * 1:hard gate
     */
    if (_tw._gate_type == 1) {
      if (_batch_size == 1) {
        /* ------to acceleratre------*/
        // moe_fw_single_stride 87ms compared to moe_fw 117ms
        moe_fw_single_stride();
      } else {
        //for-loop: perform ffn() for each gate respectively, then reorder logits according to inputs
        //faster then moe_fw() when most of gates in a batch is the same, 105ms -> 94ms
        moe_fw_hard_gate();
      }
    } else {
      //soft moe ffn
      moe_fw();
      ++_gate_weight_offset;
    }

  } else {
    ffn();
  }
  return;
}

template <OperationType OpType_>
void MoeDecoder<OpType_>::ffn() {
  /* ---step 0. layer_norm, add output_bias to "query"--- */
  ker_norm_layer_resual_launcher<_DataType>(
      _step_token_num, _tw._hidden_size, _stream, _p_d_cur_step_query,
      _p_d_query_buf1, _p_d_dec_wei[_weight_offset + 12],
      _p_d_dec_wei[_weight_offset + 13], _p_d_dec_wei[_weight_offset + 17],
      _max_thread_per_block, _tw._is_post_ln);

#ifdef DEBUG_RESULT
  print_vec(_p_d_query_buf1, "ffn ln(head): ", 5);
  print_vec(_p_d_query_buf1 + _step_token_num * _tw._hidden_size - 5,
            "ffn ln(tail): ", 5);
#endif

  /* ---step 1. first ffn layer--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._inner_size, _step_token_num,
      _tw._hidden_size, &_type_one, _p_d_dec_wei[_weight_offset + 14], _AType,
      _tw._inner_size, _p_d_query_buf1, _BType, _tw._hidden_size, &_type_zero,
      _p_d_query_buf2, _CType, _tw._inner_size, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  if (_tw._use_gelu) {
    ker_bias_gelu_launcher<_DataType>(
        _step_token_num, _max_thread_per_block, _stream, _p_d_query_buf2,
        _p_d_dec_wei[_weight_offset + 15], _tw._inner_size);
  } else {
    ker_bias_relu_launcher<_DataType>(
        _step_token_num, _max_thread_per_block, _stream, _p_d_query_buf2,
        _p_d_dec_wei[_weight_offset + 15], _tw._inner_size);
  }

  /* ---step 2. second ffn layer--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size, _step_token_num,
      _tw._inner_size, &_type_one, _p_d_dec_wei[_weight_offset + 16], _AType,
      _tw._hidden_size, _p_d_query_buf2, _BType, _tw._inner_size, &_type_one,
      _p_d_cur_step_query, _CType, _tw._hidden_size, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  return;
}

template <OperationType OpType_>
void MoeDecoder<OpType_>::set_hard_gates_ptr(int* hard_gates,
                                             std::set<int>* gate_sets,
                                             int* p_d_hard_gates) {
  _h_hard_gates = hard_gates;
  _gate_sets = gate_sets;
  _p_d_hard_gates = p_d_hard_gates;
}

template <OperationType OpType_>
void MoeDecoder<OpType_>::moe_fw_hard_gate() {
  /* ---step 0. layer_norm --- */
  //_p_d_cur_step_query: [beam_size*batch_size,hidden_dim]
  //_p_d_query_buf1: [beam_size*batch_size,hidden_dim]
  //_p_d_moe_input_buf: [beam_size*batch_size,hidden_dim]
  //_p_d_moe_input_buf_tmp: [beam_size*cur_gate_size,hidden_dim]
  //_p_d_moe_inner_buf: [beam_size*batch_size,inner_dim]
  ker_norm_layer_prepost_launcher<_DataType>(
      _step_token_num, _tw._hidden_size, _stream, _p_d_cur_step_query,
      _p_d_query_buf1, _p_d_dec_wei[_weight_offset + 12],
      _p_d_dec_wei[_weight_offset + 13], _max_thread_per_block,
      _tw._is_post_ln);

  // used for reorder ouptut of each gate
  // double pointer
  int cursor_p = 0;
  _DataType* _p_d_moe_input_buf_tmp;

  int* _p_d_cur_gate_indexs = _p_d_hard_gates + 2 * _max_batch_size;
  int sizes_index = _max_batch_size;

  for (auto it = _gate_sets->begin(); it != _gate_sets->end(); it++) {
    int cur_gate_size = _h_hard_gates[sizes_index];
    
    // _p_d_moe_input_buf_tmp: [beam_size*cur_gate_size,hidden_dim]
    // pointer of _p_d_moe_input_buf_tmp each gate will accumlate for each gate sequence
    _p_d_moe_input_buf_tmp =
        _p_d_moe_input_buf + cursor_p * _tw._beam_size * _tw._hidden_size;

    int expert_id = *it;

    int ffn1_weight_offset = _tw._inner_size * _tw._hidden_size * expert_id;
    int ffn1_bias_offset = _tw._inner_size * expert_id;

    int ffn2_weight_offset = _tw._inner_size * _tw._hidden_size * expert_id;
    int ffn2_bias_offset = _tw._hidden_size * expert_id;

    /* ---step 1. reorder batch-inputs according to gate--- */
    ker_hard_gate_reorder_pre_launcher(
        _p_d_query_buf1, _stream, cur_gate_size, _p_d_cur_gate_indexs,
        _p_d_moe_input_buf_tmp, _max_step_token_num, _tw._beam_size,
        _max_thread_per_block, _tw._hidden_size, _step_token_num);

    /* ---step 2. first ffn layer--- */
    CHECK_GPU_ERROR(cublasGemmEx(
        _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._inner_size,
        cur_gate_size * _tw._beam_size, _tw._hidden_size, &_type_one,
        _p_d_dec_wei[_weight_offset + 14] + ffn1_weight_offset, _AType,
        _tw._inner_size, _p_d_moe_input_buf_tmp, _BType, _tw._hidden_size,
        &_type_zero, _p_d_moe_inner_buf, _CType, _tw._inner_size, _computeType,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    if (_tw._use_gelu) {
      ker_bias_gelu_launcher<_DataType>(
          _tw._beam_size * cur_gate_size, _max_thread_per_block, _stream,
          _p_d_moe_inner_buf,
          _p_d_dec_wei[_weight_offset + 15] + ffn1_bias_offset, _tw._inner_size);
    } else {
      ker_bias_relu_launcher<_DataType>(
          _tw._beam_size * cur_gate_size, _max_thread_per_block, _stream,
          _p_d_moe_inner_buf,
          _p_d_dec_wei[_weight_offset + 15] + ffn1_bias_offset, _tw._inner_size);
    }

    /* ---step 3. second ffn layer--- */
    CHECK_GPU_ERROR(cublasGemmEx(
        _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size,
        _tw._beam_size * cur_gate_size, _tw._inner_size, &_type_one,
        _p_d_dec_wei[_weight_offset + 16] + ffn2_weight_offset, _AType,
        _tw._hidden_size, _p_d_moe_inner_buf, _BType, _tw._inner_size,
        &_type_zero, _p_d_moe_input_buf_tmp, _CType, _tw._hidden_size,
        _computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    cursor_p += cur_gate_size;
    _p_d_cur_gate_indexs = _p_d_cur_gate_indexs + cur_gate_size;
    sizes_index++;
  }

  /* ---step 4. reorder output of different gate--- */
  // 1. add ffn2 bias
  // 2. reorder ffn result(_p_d_moe_input_buf)  according to input
  ker_hard_gate_reorder_post_launcher(
      _stream, _p_d_moe_input_buf, _p_d_cur_step_query, _tw._beam_size,
      _max_thread_per_block, _tw._hidden_size,
      _p_d_hard_gates + 2 * _max_batch_size, _batch_size,
      _p_d_dec_wei[_weight_offset + 17], _p_d_hard_gates);
}

template <OperationType OpType_>
void MoeDecoder<OpType_>::moe_fw_single_stride() {
  //the same with ffn except ffn_weight
  /* ---step 0. layer_norm, add output_bias to "query"--- */
  int expert_id = _h_hard_gates[0];

  int ffn1_weight_offset = _tw._inner_size * _tw._hidden_size * expert_id;
  int ffn1_bias_offset = _tw._inner_size * expert_id;

  int ffn2_weight_offset = _tw._inner_size * _tw._hidden_size * expert_id;
  int ffn2_bias_offset = _tw._hidden_size * expert_id;

  ker_norm_layer_resual_launcher<_DataType>(
      _step_token_num, _tw._hidden_size, _stream, _p_d_cur_step_query,
      _p_d_query_buf1, _p_d_dec_wei[_weight_offset + 12],
      _p_d_dec_wei[_weight_offset + 13],
      _p_d_dec_wei[_weight_offset + 17] + ffn2_bias_offset,
      _max_thread_per_block, _tw._is_post_ln);

  /* ---step 1. first ffn layer--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._inner_size, _step_token_num,
      _tw._hidden_size, &_type_one,
      _p_d_dec_wei[_weight_offset + 14] + ffn1_weight_offset, _AType,
      _tw._inner_size, _p_d_query_buf1, _BType, _tw._hidden_size, &_type_zero,
      _p_d_query_buf2, _CType, _tw._inner_size, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  if (_tw._use_gelu) {
    ker_bias_gelu_launcher<_DataType>(
        _step_token_num, _max_thread_per_block, _stream, _p_d_query_buf2,
        _p_d_dec_wei[_weight_offset + 15] + ffn1_bias_offset, _tw._inner_size);
  } else {
    ker_bias_relu_launcher<_DataType>(
        _step_token_num, _max_thread_per_block, _stream, _p_d_query_buf2,
        _p_d_dec_wei[_weight_offset + 15] + ffn1_bias_offset, _tw._inner_size);
  }

  /* ---step 2. second ffn layer--- */
  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size, _step_token_num,
      _tw._inner_size, &_type_one,
      _p_d_dec_wei[_weight_offset + 16] + ffn2_weight_offset, _AType,
      _tw._hidden_size, _p_d_query_buf2, _BType, _tw._inner_size, &_type_one,
      _p_d_cur_step_query, _CType, _tw._hidden_size, _computeType,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

template <OperationType OpType_>
void MoeDecoder<OpType_>::moe_fw() {
  ker_norm_layer_prepost_launcher<_DataType>(
      _step_token_num, _tw._hidden_size, _stream, _p_d_cur_step_query,
      _p_d_query_buf1, _p_d_dec_wei[_weight_offset + 12],
      _p_d_dec_wei[_weight_offset + 13], _max_thread_per_block,
      _tw._is_post_ln);

  CHECK_GPU_ERROR(cublasGemmEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._expert_num_decoder, _step_token_num,
      _tw._hidden_size, &_type_one, _p_d_dec_gate_wei[_gate_weight_offset],
      _AType, _tw._expert_num_decoder, _p_d_query_buf1, _BType,
      _tw._hidden_size, &_type_zero, _p_d_gate, _CType, _tw._expert_num_decoder,
      _computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  ker_softmax_topk_router_launcher<_DataType>(
      _step_token_num, _tw._expert_num_decoder, _max_step_token_num,
      _tw._moe_topk_decoder, _stream, _p_d_gate, _p_d_score_routed,
      _p_d_expert_id_routed);

  ker_reorder_tokens_launcher<_DataType>(
      _step_token_num, _tw._expert_num_decoder, _max_step_token_num,
      _tw._hidden_size, _max_thread_per_block, _stream, _p_d_query_buf1,
      _p_d_score_routed, _p_d_moe_input_buf);

  CHECK_GPU_ERROR(cublasGemmStridedBatchedEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._inner_size, _step_token_num,
      _tw._hidden_size, &_type_one, _p_d_dec_wei[_weight_offset + 14], _AType,
      _tw._inner_size, _tw._hidden_size * _tw._inner_size, _p_d_moe_input_buf,
      _BType, _tw._hidden_size, _tw._hidden_size * _max_step_token_num,
      &_type_zero, _p_d_moe_inner_buf, _CType, _tw._inner_size,
      _tw._inner_size * _max_step_token_num, _tw._expert_num_decoder,
      _computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  if (_tw._use_gelu) {
    ker_strided_bias_gelu_launcher<_DataType>(
        _step_token_num, _tw._expert_num_decoder, _max_step_token_num,
        _tw._inner_size, _max_thread_per_block, _stream, _p_d_moe_inner_buf,
        _p_d_dec_wei[_weight_offset + 15]);
  } else {
    ker_strided_bias_relu_launcher<_DataType>(
        _step_token_num, _tw._expert_num_decoder, _max_step_token_num,
        _tw._inner_size, _max_thread_per_block, _stream, _p_d_moe_inner_buf,
        _p_d_dec_wei[_weight_offset + 15]);
  }

  CHECK_GPU_ERROR(cublasGemmStridedBatchedEx(
      _hd, CUBLAS_OP_N, CUBLAS_OP_N, _tw._hidden_size, _step_token_num,
      _tw._inner_size, &_type_one, _p_d_dec_wei[_weight_offset + 16], _AType,
      _tw._hidden_size, _tw._hidden_size * _tw._inner_size, _p_d_moe_inner_buf,
      _BType, _tw._inner_size, _tw._inner_size * _max_step_token_num,
      &_type_zero, _p_d_moe_input_buf, _CType, _tw._hidden_size,
      _tw._hidden_size * _max_step_token_num, _tw._expert_num_decoder,
      _computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  ker_bias_redirect_residual_launcher<_DataType>(
      _tw._hidden_size, _max_step_token_num, _tw._moe_topk_decoder,
      _step_token_num, _max_thread_per_block, _stream, _p_d_moe_input_buf,
      _p_d_dec_wei[_weight_offset + 17], _p_d_score_routed,
      _p_d_expert_id_routed, _p_d_cur_step_query);
}

template <OperationType OpType_>
bool MoeDecoder<OpType_>::sample() {
  CHECK_GPU_ERROR(
      cudaMemsetAsync(_p_d_sample_unfinished, 0, sizeof(int), _stream));
  /* --- Sample new tokens from logits --- */
  if (_tw._sampling_method == "topk") {
    ker_topk_sample_launcher<_DataType>(
        _batch_size, (_cur_step + 1), _tw._max_step, 1, _max_thread_per_block,
        _stream, _p_d_logit_buf, _p_d_trg_emb_wei[6], _p_d_alive_seq,
        _p_d_alive_seq_buf, _tw._trg_vocab_size, _tw._topk,
        _p_d_sample_unfinished, _p_d_curandstate, _tw._end_id);
  } else {
    ker_topp_sample_launcher<_DataType>(
        _batch_size, (_cur_step + 1), _tw._max_step, 1, _max_thread_per_block,
        _stream, _p_d_logit_buf, _p_d_trg_emb_wei[6], _p_d_alive_seq,
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
bool MoeDecoder<OpType_>::beam_search() {
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
    if (_h_can_num_batch < _cub_sort_buffer_bytes / 160) {
      CHECK_GPU_ERROR(cub::DeviceRadixSort::SortPairsDescending(
          (float*)_p_d_logit_buf, _cub_sort_buffer_bytes, _p_d_can_score,
          _p_d_can_score, _p_d_can_idx, _p_d_can_idx, _h_can_num_batch, 0,
          sizeof(float) * 8, _stream));
    } else {
      thrust::sort_by_key(thrust::cuda::par.on(_stream), _p_d_can_score,
                          _p_d_can_score + _h_can_num_batch, _p_d_can_idx,
                          thrust::greater<float>());
    }
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
    ker_refresh_cache_launcher<_DataType>(
        _tw._n_dec_layer * (_cur_step + 1), _step_token_num * 2,
        _max_thread_per_block, _stream, _p_d_can_num + 1, _p_d_can_idx,
        _p_d_self_k_bgeem1[0], _p_d_self_v_bgeem1[0], _p_d_self_k_bgeem2[0],
        _p_d_self_v_bgeem2[0], _layer_size_self_k, _tw._beam_size,
        _tw._dim_per_head, _tw._head_num, _tw._trg_vocab_size, _cur_step,
        _tw._max_step, _tw._diverse_lambda != 0, _tw._end_id);
    _DataType** ftmp = _p_d_self_k_bgeem2;
    _p_d_self_k_bgeem2 = _p_d_self_k_bgeem1;
    _p_d_self_k_bgeem1 = ftmp;
    ftmp = _p_d_self_v_bgeem2;
    _p_d_self_v_bgeem2 = _p_d_self_v_bgeem1;
    _p_d_self_v_bgeem1 = ftmp;
  }
  return false;
}

/**
Logits bias and softmax.
Select rough topk candidate for every batch item.
Record the candidate's beam_id, vocab_id and probability
*/
template <OperationType OpType_>
void MoeDecoder<OpType_>::update_new_seq_probs() {
  CHECK_GPU_ERROR(cudaMemsetAsync(_p_d_can_num, 0, sizeof(int), _stream));

  select_beam_rough_topk_launcher(
      _p_d_logit_buf, _p_d_trg_emb_wei[6], _p_d_alive_seq_probs,
      _p_d_alive_seq_score, _p_d_alive_seq, _p_d_can_idx, _p_d_can_score,
      _p_d_can_num, _tw._trg_vocab_size, _tw._max_step,
      _h_length_norm[_cur_step], _cur_step, _step_token_num,
      _max_thread_per_block, _stream, _tw._beam_size, _tw._diverse_lambda,
      _tw._end_id);
  thrust::exclusive_scan(thrust::cuda::par.on(_stream), _p_d_can_num + 1,
                         _p_d_can_num + 1 + _step_token_num, _p_d_can_num + 1);
  return;
}

template <OperationType OpType_>
bool MoeDecoder<OpType_>::topk_greedy_search() {
  _tw._diverse_lambda = 0;
  if (_cur_step == 0) {
    return beam_search();
  }

  CHECK_GPU_ERROR(
      cudaMemsetAsync(_p_d_sample_unfinished, 0, sizeof(int), _stream));
  /* --- Sample new tokens from logits --- */
  ker_topk_sample_launcher<_DataType>(
      _step_token_num, (_cur_step + 1), _tw._max_step, 1, _max_thread_per_block,
      _stream, _p_d_logit_buf, _p_d_trg_emb_wei[6], _p_d_alive_seq,
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

template class MoeDecoder<OperationType::FP16>;
template class MoeDecoder<OperationType::FP32>;

}  // namespace cuda
}  // namespace lightseq
