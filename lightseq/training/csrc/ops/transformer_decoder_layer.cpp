#include "transformer_decoder_layer.h"

#include "context.h"
#include "kernels.h"

template <typename T>
TransformerDecoderLayer<T>::TransformerDecoderLayer(
    int layer_id, int max_batch_tokens, int max_seq_len, int hidden_size,
    int num_heads, int intermediate_size, float attn_prob_dropout_ratio,
    float activation_dropout_ratio, float hidden_output_dropout_ratio,
    bool pre_or_postLayerNorm)
    : _layer_id(layer_id),
      _max_batch_tokens(max_batch_tokens),
      _max_seq_len(max_seq_len),
      _hidden_size(hidden_size),
      _heads(num_heads),
      _intermediate_size(intermediate_size),
      _training(true),
      _predict(false),
      _pre_or_postLayerNorm(pre_or_postLayerNorm),
      // >>> decoder self attn layer
      _attn_ln(typename Normalize_Layer<T>::Config(hidden_size, false),
               _max_batch_tokens),
      _qkv_linear(
          typename FeedForward<T>::Config(3 * hidden_size, hidden_size)),
      _attn_scores(typename StridedBatchGemm<T>::Config(
          (T(1.0) / T(sqrt(_hidden_size / _heads))), T(0.0), CUBLAS_OP_T,
          CUBLAS_OP_N)),
      _softmax(typename Softmax<T>::Config(num_heads)),
      _attn_prob_dropout(typename Dropout<T>::Config(attn_prob_dropout_ratio),
                         _max_batch_tokens * _heads * _max_seq_len),
      _attn_context(typename StridedBatchGemm<T>::Config(
          T(1.0), T(0.0), CUBLAS_OP_N, CUBLAS_OP_N)),
      _attn_out_linear(
          typename FeedForward<T>::Config(hidden_size, hidden_size)),
      _attn_dropout(typename Dropout<T>::Config(hidden_output_dropout_ratio),
                    _max_batch_tokens * _hidden_size),
      // >>> decoder enc-dec attn layer
      _encdec_attn_ln(typename Normalize_Layer<T>::Config(hidden_size, false),
                      _max_batch_tokens),
      _encdec_q_linear(
          typename FeedForward<T>::Config(hidden_size, hidden_size)),
      _encdec_kv_linear(
          typename FeedForward<T>::Config(2 * hidden_size, hidden_size)),
      _encdec_attn_scores(typename StridedBatchGemm<T>::Config(
          (T(1.0) / T(sqrt(_hidden_size / _heads))), T(0.0), CUBLAS_OP_T,
          CUBLAS_OP_N)),
      _encdec_softmax(typename Softmax<T>::Config(num_heads)),
      _encdec_attn_prob_dropout(
          typename Dropout<T>::Config(attn_prob_dropout_ratio),
          _max_batch_tokens * _heads * _max_seq_len),
      _encdec_attn_context(typename StridedBatchGemm<T>::Config(
          T(1.0), T(0.0), CUBLAS_OP_N, CUBLAS_OP_N)),
      _encdec_attn_out_linear(
          typename FeedForward<T>::Config(hidden_size, hidden_size)),
      _encdec_attn_dropout(
          typename Dropout<T>::Config(hidden_output_dropout_ratio),
          _max_batch_tokens * _hidden_size),
      // >>> decoder ffn layer
      _ffn_ln(typename Normalize_Layer<T>::Config(hidden_size, false),
              _max_batch_tokens),
      _ff1(typename FeedForward<T>::Config(_intermediate_size, hidden_size)),
      _ffn_activation_dropout(
          typename Dropout<T>::Config(activation_dropout_ratio),
          _max_batch_tokens * _intermediate_size),
      _ff2(typename FeedForward<T>::Config(hidden_size, _intermediate_size)),
      _ffn_dropout(typename Dropout<T>::Config(hidden_output_dropout_ratio),
                   _max_batch_tokens * _hidden_size) {
  assert(_hidden_size % _heads == 0);
  allocate_buffer();
  _shared_nlayer += 1;
}

template <typename T>
TransformerDecoderLayer<T>::~TransformerDecoderLayer() {
  free_memory();
}

/*------forward------*/
template <typename T>
void TransformerDecoderLayer<T>::self_attn_layer_fw(const T *input_ptr,
                                                    T *output_ptr, T *buffer,
                                                    std::vector<T *> &cache) {
  // size of buffer: [batch_size, trg_seq_len, hidden_size]
  T *q_tf_ptr = _qkv_ptr;
  T *k_tf_ptr = q_tf_ptr + _batch_dim;
  T *v_tf_ptr = k_tf_ptr + _batch_dim;
  int from_len = _predict ? 1 : _trg_seq_len;
  int to_len = _predict ? _step + 1 : _trg_seq_len;
  int batch_size = _predict ? _batch_size * _trg_seq_len : _batch_size;
  int batch_heads = _predict ? _batch_heads * _trg_seq_len : _batch_heads;
  if (_pre_or_postLayerNorm) {
    _attn_ln.Forward(_gemmQKV_inp_ptr, input_ptr, _attn_nw_ptr, _attn_nb_ptr,
                     _batch_tokens, _stream);
  }

  const T *gemmQKV_inp_ptr =
      _pre_or_postLayerNorm ? _gemmQKV_inp_ptr : input_ptr;
  _qkv_linear.Forward(_batch_tokens, gemmQKV_inp_ptr, _attn_qkvw_ptr, buffer,
                      _cublasHandle);
  launch_bias_add_transform_20314<T>(q_tf_ptr, buffer, _attn_qkvb_ptr,
                                     batch_size, from_len, 3, _heads,
                                     _hidden_size / _heads, _stream);
  if (_predict) {
    launch_concat3_dim1(cache[2], k_tf_ptr, cache[0], batch_heads,
                        _hidden_size / _heads, _step, 1, _stream);
    launch_concat3_dim1(cache[3], v_tf_ptr, cache[1], batch_heads,
                        _hidden_size / _heads, _step, 1, _stream);
    k_tf_ptr = cache[0];
    v_tf_ptr = cache[1];
  }
  // attention scores, q*k
  _attn_scores.Forward(batch_heads, _soft_out_ptr, k_tf_ptr, q_tf_ptr,
                       _cublasHandle);
  // Softmax + Mask
  _softmax.Forward(_soft_out_ptr, nullptr, batch_size, from_len, to_len,
                   _stream, _predict ? false : true);

  // attn prob dropout.
  _attn_prob_dropout.dropout(_attn_score_ptr, _soft_out_ptr,
                             batch_heads * from_len * to_len, _stream);
  // attention context, score * v
  _attn_context.Forward(batch_heads, buffer, v_tf_ptr, _attn_score_ptr,
                        _cublasHandle);

  // [b, nh, s, ad] -> [b, s, nh, ad]
  launch_transform4d_0213<T>(_attn_output_ptr, buffer, batch_size, from_len,
                             _hidden_size, _heads, 1, _stream);

  _attn_out_linear.Forward(_batch_tokens, _attn_output_ptr, _attn_ow_ptr,
                           output_ptr, _cublasHandle);
  _attn_dropout.bias_dropout_residual(output_ptr, output_ptr, input_ptr,
                                      _attn_ob_ptr, _batch_tokens, _hidden_size,
                                      _stream);
  if (!_pre_or_postLayerNorm) {
    // in-place ln since ln-input will not be used in post-ln mode
    _attn_ln.Forward(output_ptr, output_ptr, _attn_nw_ptr, _attn_nb_ptr,
                     _batch_tokens, _stream);
  }
}

template <typename T>
void TransformerDecoderLayer<T>::encdec_kv_fw(const T *enc_output_ptr) {
  allocate_encdec_kv_memory();
  _encdec_kv_linear.Forward(_batch_size * _src_seq_len, enc_output_ptr,
                            _encdec_attn_kvw_ptr, _shared_grad_encdec_kv_ptr,
                            _cublasHandle);
  // [batch_size, src_seq_len, n_dec_layer * 2, hidden_size] ->
  // [n_dec_layer * 2, batch_size, nhead, src_seq_len, head_dim]
  launch_bias_add_transform_20314<T>(
      _predict ? _shared_infer_encdec_kv_ptr : _shared_encdec_kv_ptr,
      _shared_grad_encdec_kv_ptr, _encdec_attn_kvb_ptr, _batch_size,
      _src_seq_len, _shared_nlayer * 2, _heads, _hidden_size / _heads, _stream);
}

template <typename T>
void TransformerDecoderLayer<T>::encdec_kv_bw(const T *enc_output_ptr,
                                              T *grad_enc_output_ptr) {
  // [n_dec_layer * 2, batch_size, nhead, src_seq_len, head_dim] ->
  // [batch_size, src_seq_len, n_dec_layer * 2, hidden_size] ->
  launch_transform4d_0213<T>(_shared_encdec_kv_ptr, _shared_grad_encdec_kv_ptr,
                             _batch_size, _src_seq_len, _hidden_size, _heads,
                             _shared_nlayer * 2, _stream);
  _encdec_kv_linear.Backward(
      _batch_size * _src_seq_len, _shared_encdec_kv_ptr, enc_output_ptr,
      _encdec_attn_kvw_ptr, _grad_encdec_attn_kvw_ptr,
      _grad_encdec_attn_kvb_ptr, _cublasHandle, _stream, grad_enc_output_ptr);
}

template <typename T>
void TransformerDecoderLayer<T>::encdec_attn_layer_fw(const T *input_ptr,
                                                      const T *enc_mask_ptr,
                                                      T *output_ptr,
                                                      T *buffer) {
  // size of buffer: [batch_size, trg_seq_len, hidden_size]
  if (_pre_or_postLayerNorm) {
    _encdec_attn_ln.Forward(_gemmQ_inp_ptr, input_ptr, _encdec_attn_nw_ptr,
                            _encdec_attn_nb_ptr, _batch_tokens, _stream);
  }
  _encdec_q_linear.Forward(_batch_tokens, _gemmQ_inp_ptr, _encdec_attn_qw_ptr,
                           buffer, _cublasHandle);
  // query: [batch_size, trg_seq_len, hidden_size] ->
  // [batch_size, nhead, trg_seq_len, head_dim]
  launch_bias_add_transform_20314<T>(_encdec_q_ptr, buffer, _encdec_attn_qb_ptr,
                                     _batch_size, _trg_seq_len, 1, _heads,
                                     _hidden_size / _heads, _stream);

  // attention scores, q*k
  // key: [batch_size, nhead, src_seq_len, head_dim]
  const T *encdec_kv_ptr =
      _predict ? _shared_infer_encdec_kv_ptr : _shared_encdec_kv_ptr;
  const T *encdec_k_ptr =
      encdec_kv_ptr + _layer_id * 2 * _batch_size * _src_seq_len * _hidden_size;
  // score: [batch_size, nhead, trg_seq_len, src_seq_len]
  _encdec_attn_scores.Forward(_batch_heads, _encdec_soft_out_ptr, encdec_k_ptr,
                              _encdec_q_ptr, _cublasHandle);

  // Softmax + Mask
  _encdec_softmax.Forward(_encdec_soft_out_ptr, enc_mask_ptr, _batch_size,
                          _trg_seq_len, _src_seq_len, _stream);

  // attn prob dropout.
  _encdec_attn_prob_dropout.dropout(
      _encdec_attn_score_ptr, _encdec_soft_out_ptr,
      _batch_heads * _trg_seq_len * _src_seq_len, _stream);

  // attention context, score * v
  // value: [batch_size, nhead, src_seq_len, head_dim]
  const T *encdec_v_ptr = encdec_kv_ptr + (_layer_id * 2 + 1) * _batch_size *
                                              _src_seq_len * _hidden_size;
  _encdec_attn_context.Forward(_batch_heads, buffer, encdec_v_ptr,
                               _encdec_attn_score_ptr, _cublasHandle);

  // [batch_size, nhead, trg_seq_len, head_dim] ->
  // [batch_size, trg_seq_len, hidden_size]
  launch_transform4d_0213<T>(_encdec_attn_output_ptr, buffer, _batch_size,
                             _trg_seq_len, _hidden_size, _heads, 1, _stream);

  _encdec_attn_out_linear.Forward(_batch_tokens, _encdec_attn_output_ptr,
                                  _encdec_attn_ow_ptr, output_ptr,
                                  _cublasHandle);

  _encdec_attn_dropout.bias_dropout_residual(output_ptr, output_ptr, input_ptr,
                                             _encdec_attn_ob_ptr, _batch_tokens,
                                             _hidden_size, _stream);
  if (!_pre_or_postLayerNorm) {
    // in-place ln since ln-input will not be used in post-ln mode
    _encdec_attn_ln.Forward(output_ptr, output_ptr, _encdec_attn_nw_ptr,
                            _encdec_attn_nb_ptr, _batch_tokens, _stream);
  }
}

template <typename T>
void TransformerDecoderLayer<T>::ffn_layer_fw(T *inp_ptr, T *out_ptr) {
  // save _ff1_inp_ptr, _relu_inp_ptr, _ff2_inp_ptr for backward
  if (_pre_or_postLayerNorm) {
    _ffn_ln.Forward(_ff1_inp_ptr, inp_ptr, _ffn_nw_ptr, _ffn_nb_ptr,
                    _batch_tokens, _stream);
  }
  _ff1.Forward(_batch_tokens, _ff1_inp_ptr, _inter_w_ptr, _relu_inp_ptr,
               _cublasHandle);

  _ffn_activation_dropout.bias_relu_dropout(_ff2_inp_ptr, _relu_inp_ptr,
                                            _inter_b_ptr, _batch_tokens,
                                            _intermediate_size, _stream);

  _ff2.Forward(_batch_tokens, _ff2_inp_ptr, _output_w_ptr, out_ptr,
               _cublasHandle);

  _ffn_dropout.bias_dropout_residual(out_ptr, out_ptr, inp_ptr, _output_b_ptr,
                                     _batch_tokens, _hidden_size, _stream);

  if (!_pre_or_postLayerNorm) {
    // in-place ln since ln-input will not be used in post-ln mode
    _ffn_ln.Forward(out_ptr, out_ptr, _ffn_nw_ptr, _ffn_nb_ptr, _batch_tokens,
                    _stream);
  }
}

template <typename T>
void TransformerDecoderLayer<T>::Forward(const T *dec_input_ptr,
                                         const T *enc_output_ptr,
                                         const T *enc_mask_ptr,
                                         T *dec_output_ptr,
                                         std::vector<T *> &cache) {
  _stream = Context::Instance().get_stream();
  _cublasHandle = Context::Instance().get_cublashandle();
  if (_predict && _step == 0 && _layer_id == 0) {
    _shared_infer_encdec_kv_ptr = cache[4];
    encdec_kv_fw(enc_output_ptr);
  }
  if (!_predict && _layer_id == 0) {
    encdec_kv_fw(enc_output_ptr);
  }

  T *buffer = _shared_buffer_ptr;  // 3 * _batch_dim
  // _batch_dim
  T *encdec_attn_inp_ptr =
      _pre_or_postLayerNorm ? buffer + 3 * _batch_dim : _gemmQ_inp_ptr;
  // _batch_dim
  T *ffn_inp_ptr =
      _pre_or_postLayerNorm ? buffer + 4 * _batch_dim : _ff1_inp_ptr;

  self_attn_layer_fw(dec_input_ptr, encdec_attn_inp_ptr, buffer, cache);

  encdec_attn_layer_fw(encdec_attn_inp_ptr, enc_mask_ptr, ffn_inp_ptr, buffer);

  ffn_layer_fw(ffn_inp_ptr, dec_output_ptr);
  // cudaStreamSynchronize(_stream);
  // CHECK_GPU_ERROR(cudaGetLastError());
}

/*------backward------*/
template <typename T>
void TransformerDecoderLayer<T>::self_attn_layer_bw(const T *input_ptr,
                                                    const T *output_ptr,
                                                    const T *grad_output_ptr,
                                                    T *grad_input_ptr,
                                                    T *buffer) {
  cudaStream_t streams[2] = {_stream, _stream};
  const T *q_tf_ptr = _qkv_ptr;
  const T *k_tf_ptr = q_tf_ptr + _batch_dim;
  const T *v_tf_ptr = k_tf_ptr + _batch_dim;
  // batch_dim = batch_size * seq_len * hidden_size
  // buffer size: batch_dim * 3 + max(batch_dim * 3,
  //     batch_size * head_num * seq_len * seq_len)
  T *grad_residual_ptr = buffer;
  buffer += _batch_dim;

  T *grad_input_buf_ptr = buffer;  // batch_dim
  T *grad_qkv_5d_ptr = buffer;     // batch_dim * 3
  buffer += 3 * _batch_dim;

  T *grad_qkv_4d_ptr = buffer;   // batch_dim * 3
  T *grad_softmax_ptr = buffer;  // batch_size * head_num * seq_len * seq_len
  // buffer += max(3 * _batch_dim,
  //   batch_size * head_num * seq_len * seq_len);

  if (_pre_or_postLayerNorm) {
    _attn_dropout.d_bias_dropout_residual(grad_input_ptr, _grad_attn_ob_ptr,
                                          grad_output_ptr, _batch_tokens,
                                          _hidden_size, _stream);
  } else {
    _attn_ln.Backward(_grad_attn_nw_ptr, _grad_attn_nb_ptr, grad_residual_ptr,
                      grad_output_ptr, nullptr, output_ptr, _attn_nw_ptr,
                      _attn_nb_ptr, _batch_tokens, streams);
    _attn_dropout.d_bias_dropout_residual(grad_input_ptr, _grad_attn_ob_ptr,
                                          grad_residual_ptr, _batch_tokens,
                                          _hidden_size, _stream);
  }

  // bw of output project
  _attn_out_linear.Backward(_batch_tokens, grad_input_ptr, _attn_output_ptr,
                            _attn_ow_ptr, _grad_attn_ow_ptr, _grad_attn_ob_ptr,
                            _cublasHandle, _stream, grad_input_buf_ptr, nullptr,
                            false);
  launch_transform_0213<T>(grad_input_ptr, grad_input_buf_ptr, _batch_size,
                           _trg_seq_len, _hidden_size, _heads, _stream);

  // bw of score * v
  _attn_context.Backward(_batch_heads, grad_input_ptr, v_tf_ptr,
                         _attn_score_ptr, _cublasHandle,
                         grad_qkv_5d_ptr + 2 * _batch_dim, grad_softmax_ptr);

  _attn_prob_dropout.d_dropout(
      grad_softmax_ptr, _batch_heads * _trg_seq_len * _trg_seq_len, _stream);

  _softmax.Backward(grad_softmax_ptr, _soft_out_ptr, _batch_size, _trg_seq_len,
                    _trg_seq_len, _stream);

  // bw of q * k
  _attn_scores.Backward(_batch_heads, grad_softmax_ptr, k_tf_ptr, q_tf_ptr,
                        _cublasHandle, grad_qkv_5d_ptr + _batch_dim,
                        grad_qkv_5d_ptr);

  // [3, b, nh, s, ad] -> [b, s, 3, h]
  launch_transform4d_0213<T>(grad_qkv_4d_ptr, grad_qkv_5d_ptr, _batch_size,
                             _trg_seq_len, _hidden_size, _heads, 3, _stream);

  const T *gemmQKV_inp_ptr =
      _pre_or_postLayerNorm ? _gemmQKV_inp_ptr : input_ptr;
  _qkv_linear.Backward(_batch_tokens, grad_qkv_4d_ptr, gemmQKV_inp_ptr,
                       _attn_qkvw_ptr, _grad_attn_qkvw_ptr, _grad_attn_qkvb_ptr,
                       _cublasHandle, _stream, grad_input_buf_ptr);

  if (_pre_or_postLayerNorm) {
    _attn_ln.Backward(_grad_attn_nw_ptr, _grad_attn_nb_ptr, grad_input_ptr,
                      grad_input_buf_ptr, grad_output_ptr, gemmQKV_inp_ptr,
                      _attn_nw_ptr, _attn_nb_ptr, _batch_tokens, streams);
  } else {
    // FIXME later
    launch_fused_add2<T>(grad_input_ptr, grad_input_buf_ptr, grad_residual_ptr,
                         _batch_size, _trg_seq_len, _hidden_size, _stream);
  }
}

template <typename T>
void TransformerDecoderLayer<T>::encdec_attn_layer_bw(const T *output_ptr,
                                                      const T *grad_output_ptr,
                                                      T *grad_input_ptr,
                                                      T *buffer) {
  cudaStream_t streams[2] = {_stream, _stream};

  int offset = _batch_size * _src_seq_len * _hidden_size;
  // key: [batch_size, nhead, src_seq_len, head_dim]
  const T *k_ptr = _shared_encdec_kv_ptr + _layer_id * 2 * offset;
  // value: [batch_size, nhead, src_seq_len, head_dim]
  const T *v_ptr = _shared_encdec_kv_ptr + (_layer_id * 2 + 1) * offset;
  // key: [batch_size, nhead, src_seq_len, head_dim]
  T *grad_k_ptr = _shared_grad_encdec_kv_ptr + _layer_id * 2 * offset;
  // value: [batch_size, nhead, src_seq_len, head_dim]
  T *grad_v_ptr = _shared_grad_encdec_kv_ptr + (_layer_id * 2 + 1) * offset;

  // batch_dim = batch_size * seq_len * hidden_size
  // buffer size: batch_dim * 2 + max(batch_dim,
  //     batch_size * head_num * seq_len * seq_len)
  T *grad_residual_ptr = buffer;  // batch_dim
  buffer += _batch_dim;

  T *grad_input_buf_ptr = buffer;  // batch_dim
  T *grad_q_5d_ptr = buffer;       // batch_dim
  buffer += _batch_dim;

  T *grad_q_4d_ptr = buffer;  // batch_dim
  // batch_size * head_num * trg_seq_len * src_seq_len
  T *grad_softmax_ptr = buffer;

  if (_pre_or_postLayerNorm) {
    _encdec_attn_dropout.d_bias_dropout_residual(
        grad_input_ptr, _grad_encdec_attn_ob_ptr, grad_output_ptr,
        _batch_tokens, _hidden_size, _stream);
  } else {
    _encdec_attn_ln.Backward(_grad_encdec_attn_nw_ptr, _grad_encdec_attn_nb_ptr,
                             grad_residual_ptr, grad_output_ptr, nullptr,
                             output_ptr, _encdec_attn_nw_ptr,
                             _encdec_attn_nb_ptr, _batch_tokens, streams);
    _encdec_attn_dropout.d_bias_dropout_residual(
        grad_input_ptr, _grad_encdec_attn_ob_ptr, grad_residual_ptr,
        _batch_tokens, _hidden_size, _stream);
  }

  // bw of output project
  _encdec_attn_out_linear.Backward(
      _batch_tokens, grad_input_ptr, _encdec_attn_output_ptr,
      _encdec_attn_ow_ptr, _grad_encdec_attn_ow_ptr, _grad_encdec_attn_ob_ptr,
      _cublasHandle, _stream, grad_input_buf_ptr, nullptr, false);
  launch_transform_0213<T>(grad_input_ptr, grad_input_buf_ptr, _batch_size,
                           _trg_seq_len, _hidden_size, _heads, _stream);

  // bw of score * v
  _encdec_attn_context.Backward(_batch_heads, grad_input_ptr, v_ptr,
                                _encdec_attn_score_ptr, _cublasHandle,
                                grad_v_ptr, grad_softmax_ptr);

  _encdec_attn_prob_dropout.d_dropout(
      grad_softmax_ptr, _batch_heads * _trg_seq_len * _src_seq_len, _stream);

  _encdec_softmax.Backward(grad_softmax_ptr, _encdec_soft_out_ptr, _batch_size,
                           _trg_seq_len, _src_seq_len, _stream);

  // bw of q * k
  _encdec_attn_scores.Backward(_batch_heads, grad_softmax_ptr, k_ptr,
                               _encdec_q_ptr, _cublasHandle, grad_k_ptr,
                               grad_q_5d_ptr);

  // [3, b, nh, s, ad] -> [b, s, 3, h]
  launch_transform4d_0213<T>(grad_q_4d_ptr, grad_q_5d_ptr, _batch_size,
                             _trg_seq_len, _hidden_size, _heads, 1, _stream);

  _encdec_q_linear.Backward(_batch_tokens, grad_q_4d_ptr, _gemmQ_inp_ptr,
                            _encdec_attn_qw_ptr, _grad_encdec_attn_qw_ptr,
                            _grad_encdec_attn_qb_ptr, _cublasHandle, _stream,
                            grad_input_buf_ptr);

  if (_pre_or_postLayerNorm) {
    _encdec_attn_ln.Backward(
        _grad_encdec_attn_nw_ptr, _grad_encdec_attn_nb_ptr, grad_input_ptr,
        grad_input_buf_ptr, grad_output_ptr, _gemmQ_inp_ptr,
        _encdec_attn_nw_ptr, _encdec_attn_nb_ptr, _batch_tokens, streams);
  } else {
    // FIXME later
    launch_fused_add2<T>(grad_input_ptr, grad_input_buf_ptr, grad_residual_ptr,
                         _batch_size, _trg_seq_len, _hidden_size, _stream);
  }
}

template <typename T>
void TransformerDecoderLayer<T>::ffn_layer_bw(const T *grad_output_ptr,
                                              const T *output_ptr,
                                              T *grad_inp_ptr, T *buffer) {
  cudaStream_t streams[2] = {_stream, _stream};

  T *grad_residual_ptr = buffer;
  buffer += _batch_dim;

  T *grad_ff1_inp_ptr = buffer;
  buffer += _batch_dim;

  T *grad_ff1_out_ptr = buffer;
  // buffer += _batch_size * _trg_seq_len * _intermediate_size;

  if (_pre_or_postLayerNorm) {
    _ffn_dropout.d_bias_dropout_residual(grad_inp_ptr, _grad_output_b_ptr,
                                         grad_output_ptr, _batch_tokens,
                                         _hidden_size, _stream);
  } else {
    _ffn_ln.Backward(_grad_ffn_nw_ptr, _grad_ffn_nb_ptr, grad_residual_ptr,
                     grad_output_ptr, nullptr, output_ptr, _ffn_nw_ptr,
                     _ffn_nb_ptr, _batch_tokens, streams);
    _ffn_dropout.d_bias_dropout_residual(grad_inp_ptr, _grad_output_b_ptr,
                                         grad_residual_ptr, _batch_tokens,
                                         _hidden_size, _stream);
  }

  _ff2.Backward(_batch_tokens, grad_inp_ptr, _ff2_inp_ptr, _output_w_ptr,
                _grad_output_w_ptr, _grad_output_b_ptr, _cublasHandle, _stream,
                grad_ff1_out_ptr, nullptr, false);

  _ffn_activation_dropout.d_bias_relu_dropout(
      grad_ff1_out_ptr, _grad_inter_b_ptr, _relu_inp_ptr, _inter_b_ptr,
      _batch_tokens, _intermediate_size, _stream);

  _ff1.Backward(_batch_tokens, grad_ff1_out_ptr, _ff1_inp_ptr, _inter_w_ptr,
                _grad_inter_w_ptr, _grad_inter_b_ptr, _cublasHandle, _stream,
                grad_ff1_inp_ptr, nullptr, false);

  /* ln signature:
  grad_gamma_grad, grad_betta, grad_inp,
  grad_out, grad_residual, output, gamma, betta,
  */
  const T *add_res_ptr = _ff1_inp_ptr;
  if (_pre_or_postLayerNorm) {
    _ffn_ln.Backward(_grad_ffn_nw_ptr, _grad_ffn_nb_ptr, grad_inp_ptr,
                     grad_ff1_inp_ptr, grad_output_ptr, _ff1_inp_ptr,
                     _ffn_nw_ptr, _ffn_nb_ptr, _batch_tokens, streams);
  } else {
    launch_fused_add2<T>(grad_inp_ptr, grad_ff1_inp_ptr, grad_residual_ptr,
                         _batch_size, _trg_seq_len, _hidden_size, _stream);
  }
}

template <typename T>
void TransformerDecoderLayer<T>::Backward(
    const T *grad_dec_output_ptr, const T *dec_input_ptr,
    const T *enc_output_ptr, const T *enc_mask_ptr, const T *dec_output_ptr,
    T *grad_dec_input_ptr, T *grad_enc_output_ptr) {
  _stream = Context::Instance().get_stream();
  _cublasHandle = Context::Instance().get_cublashandle();
  /*
  buffer size needed by ffn bw:
      2 * _batch_dim + _batch_size * _seq_len * _intermediate_size
  */
  T *grad_ffn_inp_ptr = _shared_buffer_ptr;
  T *buffer = grad_ffn_inp_ptr + _batch_dim;
  ffn_layer_bw(grad_dec_output_ptr, dec_output_ptr, grad_ffn_inp_ptr, buffer);

  /*
  buffer size needed by encdec attn:
      2 * batch_dim + max(batch_dim,
      batch_size * head_num * seq_len * seq_len)
  */
  // since _relu_inp_ptr will not be used after ffn bw
  // FIXME later.
  T *grad_encdec_inp_ptr = _relu_inp_ptr;
  encdec_attn_layer_bw(_ff1_inp_ptr, grad_ffn_inp_ptr, grad_encdec_inp_ptr,
                       buffer);

  /*
  buffer size needed by attn bw:
      4 * _batch_dim + max(3 * _batch_dim,
      _batch_size * _head_num * _seq_len * _seq_len);
  */
  self_attn_layer_bw(dec_input_ptr, _gemmQ_inp_ptr, grad_encdec_inp_ptr,
                     grad_dec_input_ptr, buffer);

  if (_layer_id == 0) {
    encdec_kv_bw(enc_output_ptr, grad_enc_output_ptr);
  }
  // cudaStreamSynchronize(_stream);
  // CHECK_GPU_ERROR(cudaGetLastError());
}

template <typename T>
size_t TransformerDecoderLayer<T>::_shared_nlayer = 0;
template <typename T>
T *TransformerDecoderLayer<T>::_shared_buffer_ptr = nullptr;
template <typename T>
T *TransformerDecoderLayer<T>::_shared_encdec_kv_ptr = nullptr;
template <typename T>
T *TransformerDecoderLayer<T>::_shared_grad_encdec_kv_ptr = nullptr;
template <typename T>
T *TransformerDecoderLayer<T>::_shared_infer_encdec_kv_ptr = nullptr;

template class TransformerDecoderLayer<float>;
template class TransformerDecoderLayer<__half>;
