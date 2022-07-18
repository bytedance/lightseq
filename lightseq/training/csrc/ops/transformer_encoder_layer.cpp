#include "transformer_encoder_layer.h"

#include "context.h"
#include "kernels.h"

template <typename T>
TransformerEncoderLayer<T>::TransformerEncoderLayer(
    int layer_id, int max_batch_tokens, int max_seq_len, int hidden_size,
    int num_heads, int intermediate_size, float attn_prob_dropout_ratio,
    float activation_dropout_ratio, float hidden_output_dropout_ratio,
    bool pre_or_postLayerNorm, std::string activation_fn,
    bool mask_future_tokens)
    : _layer_id(layer_id),
      _max_batch_tokens(max_batch_tokens),
      _max_seq_len(max_seq_len),
      _hidden_size(hidden_size),
      _heads(num_heads),
      _intermediate_size(intermediate_size),
      _training(true),
      _pre_or_postLayerNorm(pre_or_postLayerNorm),
      _activation_fn(activation_fn),
      _qkv_linear(
          typename FeedForward<T>::Config(3 * hidden_size, hidden_size)),
      _attn_out_linear(
          typename FeedForward<T>::Config(hidden_size, hidden_size)),
      _attn_ln(typename Normalize_Layer<T>::Config(hidden_size, false),
               _max_batch_tokens),
      _ffn_ln(typename Normalize_Layer<T>::Config(hidden_size, false),
              _max_batch_tokens),
      _ff1(typename FeedForward<T>::Config(_intermediate_size, hidden_size)),
      _ff2(typename FeedForward<T>::Config(hidden_size, _intermediate_size)),
      _softmax(typename Softmax<T>::Config(num_heads, mask_future_tokens)),
      _attn_prob_dropout(typename Dropout<T>::Config(attn_prob_dropout_ratio),
                         _max_batch_tokens * _heads * _max_seq_len),
      _attn_dropout(typename Dropout<T>::Config(hidden_output_dropout_ratio),
                    _max_batch_tokens * _hidden_size),
      _ffn_activation_dropout(
          typename Dropout<T>::Config(activation_dropout_ratio),
          _max_batch_tokens * _intermediate_size),
      _ffn_dropout(typename Dropout<T>::Config(hidden_output_dropout_ratio),
                   _max_batch_tokens * _hidden_size),
      _attn_scores(typename StridedBatchGemm<T>::Config(
          (T(1.0) / T(sqrt(_hidden_size / _heads))), T(0.0), CUBLAS_OP_T,
          CUBLAS_OP_N)),
      _attn_context(typename StridedBatchGemm<T>::Config(
          T(1.0), T(0.0), CUBLAS_OP_N, CUBLAS_OP_N)),
      _enable_quant(false) {
  assert(_hidden_size % _heads == 0);
  allocate_mem_buffer();
}

template <typename T>
TransformerEncoderLayer<T>::~TransformerEncoderLayer() {
  free_mem_buffer();
}

template <typename T>
void TransformerEncoderLayer<T>::attn_layer_fw(const T *input_ptr,
                                               const T *input_mask_ptr,
                                               T *output_ptr, T *buffer) {
  T *q_tf_ptr = _qkv_ptr;
  T *k_tf_ptr = q_tf_ptr + _batch_dim;
  T *v_tf_ptr = k_tf_ptr + _batch_dim;

  int8_t *i8_buffer_ptr = reinterpret_cast<int8_t *>(buffer);
  int8_t *qout_ptr = i8_buffer_ptr;
  int8_t *qweight_ptr = qout_ptr + _batch_dim * 3;

  if (_enable_quant) {
    if (_pre_or_postLayerNorm) {
      _attn_ln.Forward(_gemmQKV_inp_i8_ptr, _attn_prob_dropout.get_mask(),
                       input_ptr, _attn_nw_ptr, _attn_nb_ptr,
                       _attn_qkv_cmax_ptr, _batch_tokens, _stream);

    } else {
      launch_quantize<T>(_gemmQKV_inp_i8_ptr, _attn_prob_dropout.get_mask(),
                         nullptr, input_ptr, _attn_qkv_cmax_ptr,
                         _hidden_size * _batch_tokens, 2, _stream);
    }

    launch_quantize<T>(qweight_ptr, _attn_prob_dropout.get_mask(),
                       _igemm_alpha_ptr, _attn_qkvw_ptr, _attn_qkv_cmax_ptr,
                       _hidden_size * 3 * _hidden_size, 4, _stream);

    _qkv_linear.Forward(_batch_tokens, _gemmQKV_inp_i8_ptr, qweight_ptr,
                        _igemm_alpha_ptr, _igemm_beta_ptr, qout_ptr,
                        _cublasLtHandle, _stream);

    launch_quant_bias_add_transform_20314<T>(
        q_tf_ptr, _attn_prob_dropout.get_mask(), qout_ptr, _attn_qkvb_ptr,
        _attn_qkv_cmax_ptr + 2, _batch_size, _seq_len, 3, _heads,
        _hidden_size / _heads, _stream);
  } else {
    if (_pre_or_postLayerNorm) {
      _attn_ln.Forward(_gemmQKV_inp_ptr, input_ptr, _attn_nw_ptr, _attn_nb_ptr,
                       _batch_tokens, _stream);
    }
    const T *gemmQKV_inp_ptr =
        _pre_or_postLayerNorm ? _gemmQKV_inp_ptr : input_ptr;
    _qkv_linear.Forward(_batch_tokens, gemmQKV_inp_ptr, _attn_qkvw_ptr, buffer,
                        _cublasHandle);

    launch_bias_add_transform_20314<T>(q_tf_ptr, buffer, _attn_qkvb_ptr,
                                       _batch_size, _seq_len, 3, _heads,
                                       _hidden_size / _heads, _stream);
  }

  // attention scores, q*k
  _attn_scores.Forward(_batch_heads, _soft_out_ptr, k_tf_ptr, q_tf_ptr,
                       _cublasHandle);

  // Softmax + Mask
  _softmax.Forward(_soft_out_ptr, input_mask_ptr, _batch_size, _seq_len,
                   _seq_len, _stream);

  // attn prob dropout.
  _attn_prob_dropout.dropout(_ctx_bufB_ptr, _soft_out_ptr,
                             _batch_heads * _seq_len * _seq_len, _stream);

  // attention context, score * v
  _attn_context.Forward(_batch_heads, buffer, v_tf_ptr, _ctx_bufB_ptr,
                        _cublasHandle);

  if (_enable_quant) {
    // [b, nh, s, ad] -> [b, s, nh, ad]
    launch_quant_transform4d_0213<T>(_attn_o_inp_i8_ptr,
                                     _attn_dropout.get_mask(), buffer,
                                     _attn_out_cmax_ptr, _batch_size, _seq_len,
                                     _hidden_size, _heads, 1, _stream);

    launch_quantize<T>(qweight_ptr, _attn_dropout.get_mask(), _igemm_alpha_ptr,
                       _attn_ow_ptr, _attn_out_cmax_ptr,
                       _hidden_size * _hidden_size, 4, _stream);

    _attn_out_linear.Forward(_batch_tokens, _attn_o_inp_i8_ptr, qweight_ptr,
                             _igemm_alpha_ptr, _igemm_beta_ptr, qout_ptr,
                             _cublasLtHandle, _stream);

    _attn_dropout.quant_bias_dropout_residual(
        output_ptr, qout_ptr, _attn_out_cmax_ptr, input_ptr, _attn_ob_ptr,
        _batch_tokens, _hidden_size, _stream);
  } else {
    // [b, nh, s, ad] -> [b, s, nh, ad]
    launch_transform4d_0213<T>(_attn_o_inp_ptr, buffer, _batch_size, _seq_len,
                               _hidden_size, _heads, 1, _stream);

    _attn_out_linear.Forward(_batch_tokens, _attn_o_inp_ptr, _attn_ow_ptr,
                             output_ptr, _cublasHandle);

    _attn_dropout.bias_dropout_residual(output_ptr, output_ptr, input_ptr,
                                        _attn_ob_ptr, _batch_tokens,
                                        _hidden_size, _stream);
  }

  if (!_pre_or_postLayerNorm) {
    // in-place ln since ln-input will not be used in post-ln mode
    _attn_ln.Forward(output_ptr, output_ptr, _attn_nw_ptr, _attn_nb_ptr,
                     _batch_tokens, _stream);
  }
}

template <typename T>
void TransformerEncoderLayer<T>::ffn_layer_fw(T *inp_ptr, T *out_ptr) {
  // save _ff1_inp_ptr, _relu_inp_ptr, _ff2_inp_ptr for backward
  if (_enable_quant) {
    int8_t *i8_buffer_ptr = reinterpret_cast<int8_t *>(_shared_mem_ptr);
    int8_t *qweight_ptr = i8_buffer_ptr;
    int8_t *qout_ptr = qweight_ptr + _intermediate_size * _hidden_size;
    if (_pre_or_postLayerNorm) {
      _ffn_ln.Forward(_ff1_inp_i8_ptr, _ffn_dropout.get_mask(), inp_ptr,
                      _ffn_nw_ptr, _ffn_nb_ptr, _inter_cmax_ptr, _batch_tokens,
                      _stream);
    } else {
      launch_quantize<T>(_ff1_inp_i8_ptr, _ffn_dropout.get_mask(), nullptr,
                         inp_ptr, _inter_cmax_ptr, _hidden_size * _batch_tokens,
                         2, _stream);
    }

    launch_quantize<T>(qweight_ptr, _ffn_activation_dropout.get_mask(),
                       _igemm_alpha_ptr, _inter_w_ptr, _inter_cmax_ptr,
                       _hidden_size * _intermediate_size, 4, _stream);

    _ff1.Forward(_batch_tokens, _ff1_inp_i8_ptr, qweight_ptr, _igemm_alpha_ptr,
                 _igemm_beta_ptr, _act_inp_i8_ptr, _cublasLtHandle, _stream);

    _ffn_activation_dropout.quant_bias_act_dropout(
        _ff2_inp_i8_ptr, _ffn_activation_dropout.get_mask(),
        _ffn_activation_dropout.get_mask(), _act_inp_i8_ptr, _inter_b_ptr,
        _inter_cmax_ptr + 2, _output_cmax_ptr, _batch_tokens,
        _intermediate_size, _activation_fn, _stream);

    launch_quantize<T>(qweight_ptr, _ffn_dropout.get_mask(), _igemm_alpha_ptr,
                       _output_w_ptr, _output_cmax_ptr,
                       _hidden_size * _intermediate_size, 4, _stream);
    _ff2.Forward(_batch_tokens, _ff2_inp_i8_ptr, qweight_ptr, _igemm_alpha_ptr,
                 _igemm_beta_ptr, qout_ptr, _cublasLtHandle, _stream);

    _ffn_dropout.quant_bias_dropout_residual(
        out_ptr, qout_ptr, _output_cmax_ptr, inp_ptr, _output_b_ptr,
        _batch_tokens, _hidden_size, _stream);

    if (!_pre_or_postLayerNorm) {
      // in-place ln since ln-input will not be used in post-ln mode
      _ffn_ln.Forward(out_ptr, out_ptr, _ffn_nw_ptr, _ffn_nb_ptr, _batch_tokens,
                      _stream);
    }

    return;
  }

  if (_pre_or_postLayerNorm) {
    _ffn_ln.Forward(_ff1_inp_ptr, inp_ptr, _ffn_nw_ptr, _ffn_nb_ptr,
                    _batch_tokens, _stream);
  }
  _ff1.Forward(_batch_tokens, _ff1_inp_ptr, _inter_w_ptr, _relu_inp_ptr,
               _cublasHandle);

  _ffn_activation_dropout.bias_act_dropout(
      _ff2_inp_ptr, _relu_inp_ptr, _inter_b_ptr, _batch_tokens,
      _intermediate_size, _activation_fn, _stream);

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
void TransformerEncoderLayer<T>::Forward(const T *input_ptr,
                                         const T *input_mask_ptr, T *out_ptr) {
  _stream = Context::Instance().get_stream();
  _cublasHandle = Context::Instance().get_cublashandle();
  _cublasLtHandle = Context::Instance().get_cublaslthandle();
  T *attn_buffer = _shared_mem_ptr;  // 3 * _batch_dim
  // _batch_dim
  T *ffn_inp_ptr =
      _pre_or_postLayerNorm
          ? _shared_mem_ptr +
                std::max(3 * _batch_dim, _intermediate_size * _hidden_size)
          : _ff1_inp_ptr;

  attn_layer_fw(input_ptr, input_mask_ptr, ffn_inp_ptr, attn_buffer);

  ffn_layer_fw(ffn_inp_ptr, out_ptr);
}

template <typename T>
void TransformerEncoderLayer<T>::attn_layer_bw(const T *input_ptr,
                                               const T *input_mask_ptr,
                                               const T *grad_output_ptr,
                                               T *grad_input_ptr, T *buffer) {
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

  if (_pre_or_postLayerNorm) {
    _attn_dropout.d_bias_dropout_residual(grad_input_ptr, _grad_attn_ob_ptr,
                                          grad_output_ptr, _batch_tokens,
                                          _hidden_size, _stream);
  } else {
    _attn_ln.Backward(_grad_attn_nw_ptr, _grad_attn_nb_ptr, grad_residual_ptr,
                      grad_output_ptr, nullptr, _ff1_inp_ptr, _attn_nw_ptr,
                      _attn_nb_ptr, _batch_tokens, streams);
    _attn_dropout.d_bias_dropout_residual(grad_input_ptr, _grad_attn_ob_ptr,
                                          grad_residual_ptr, _batch_tokens,
                                          _hidden_size, _stream);
  }

  if (_enable_quant) {
    _attn_o_inp_ptr = buffer;
    // T *_clipped_w_ptr = buffer + _batch_dim;
    launch_dequantize<T>(_attn_o_inp_ptr, _attn_o_inp_i8_ptr,
                         _attn_out_cmax_ptr, _batch_tokens * _hidden_size, 2,
                         _stream);

    // bw of output project
    _attn_out_linear.Backward(_batch_tokens, grad_input_ptr, _attn_o_inp_ptr,
                              _attn_ow_ptr, _grad_attn_ow_ptr,
                              _grad_attn_ob_ptr, _cublasHandle, _stream,
                              grad_input_buf_ptr, nullptr, false);

    // launch_d_cmax(_grad_attn_ow_ptr, _grad_attn_out_cmax_ptr + 1,
    //               _attn_dropout.get_mask(), _hidden_size * _hidden_size, 4,
    //               _stream);

    launch_transform_0213_dcmax<T>(grad_input_ptr, _grad_attn_out_cmax_ptr,
                                   grad_input_buf_ptr, _attn_dropout.get_mask(),
                                   _batch_size, _seq_len, _hidden_size, _heads,
                                   _stream);
  } else {
    // bw of output project
    _attn_out_linear.Backward(_batch_tokens, grad_input_ptr, _attn_o_inp_ptr,
                              _attn_ow_ptr, _grad_attn_ow_ptr,
                              _grad_attn_ob_ptr, _cublasHandle, _stream,
                              grad_input_buf_ptr, nullptr, false);
    launch_transform_0213<T>(grad_input_ptr, grad_input_buf_ptr, _batch_size,
                             _seq_len, _hidden_size, _heads, _stream);
  }

  // bw of score * v
  _attn_context.Backward(_batch_heads, grad_input_ptr, v_tf_ptr, _ctx_bufB_ptr,
                         _cublasHandle, grad_qkv_5d_ptr + 2 * _batch_dim,
                         grad_softmax_ptr);

  _attn_prob_dropout.d_dropout(grad_softmax_ptr,
                               _batch_heads * _seq_len * _seq_len, _stream);

  _softmax.Backward(grad_softmax_ptr, _soft_out_ptr, _batch_size, _seq_len,
                    _seq_len, _stream);

  // bw of q * k
  _attn_scores.Backward(_batch_heads, grad_softmax_ptr, k_tf_ptr, q_tf_ptr,
                        _cublasHandle, grad_qkv_5d_ptr + _batch_dim,
                        grad_qkv_5d_ptr);

  // [3, b, nh, s, ad] -> [b, s, 3, h]
  launch_transform4d_0213<T>(grad_qkv_4d_ptr, grad_qkv_5d_ptr, _batch_size,
                             _seq_len, _hidden_size, _heads, 3, _stream);

  if (_enable_quant) {
    T *gemmQKV_inp_ptr = _qkv_ptr;

    launch_dequantize<T>(gemmQKV_inp_ptr, _gemmQKV_inp_i8_ptr,
                         _attn_qkv_cmax_ptr, _batch_tokens * _hidden_size, 2,
                         _stream);

    _qkv_linear.Backward(_batch_tokens, grad_qkv_4d_ptr, gemmQKV_inp_ptr,
                         _attn_qkvw_ptr, _grad_attn_qkvw_ptr,
                         _grad_attn_qkvb_ptr, _cublasHandle, _stream,
                         grad_input_buf_ptr);

    // launch_d_cmax(_grad_attn_qkvw_ptr, _grad_attn_qkv_cmax_ptr + 1,
    //               _attn_prob_dropout.get_mask(),
    //               _hidden_size * _hidden_size * 3, 4, _stream);

    // use_mean should be True when enable_quant, because we can't get
    // layer norm output before clip
    if (_pre_or_postLayerNorm) {
      _attn_ln.Backward(_grad_attn_nw_ptr, _grad_attn_nb_ptr, grad_input_ptr,
                        _grad_attn_qkv_cmax_ptr, grad_input_buf_ptr,
                        grad_output_ptr, gemmQKV_inp_ptr, _attn_nw_ptr,
                        _attn_nb_ptr, _attn_prob_dropout.get_mask(),
                        _batch_tokens, streams);
    } else {
      launch_d_cmax(grad_input_buf_ptr, _grad_attn_qkv_cmax_ptr,
                    _attn_prob_dropout.get_mask(), _batch_dim, 2, _stream);

      // FIXME later
      launch_fused_add2<T>(grad_input_ptr, grad_input_buf_ptr,
                           grad_residual_ptr, _batch_size, _seq_len,
                           _hidden_size, _stream);
    }
  } else {
    const T *gemmQKV_inp_ptr =
        _pre_or_postLayerNorm ? _gemmQKV_inp_ptr : input_ptr;
    _qkv_linear.Backward(_batch_tokens, grad_qkv_4d_ptr, gemmQKV_inp_ptr,
                         _attn_qkvw_ptr, _grad_attn_qkvw_ptr,
                         _grad_attn_qkvb_ptr, _cublasHandle, _stream,
                         grad_input_buf_ptr);
    if (_pre_or_postLayerNorm) {
      _attn_ln.Backward(_grad_attn_nw_ptr, _grad_attn_nb_ptr, grad_input_ptr,
                        grad_input_buf_ptr, grad_output_ptr, gemmQKV_inp_ptr,
                        _attn_nw_ptr, _attn_nb_ptr, _batch_tokens, streams);
    } else {
      // FIXME later
      launch_fused_add2<T>(grad_input_ptr, grad_input_buf_ptr,
                           grad_residual_ptr, _batch_size, _seq_len,
                           _hidden_size, _stream);
    }
  }
}

template <typename T>
void TransformerEncoderLayer<T>::ffn_layer_bw(const T *grad_output_ptr,
                                              const T *output_ptr,
                                              T *grad_inp_ptr, T *buffer) {
  cudaStream_t streams[2] = {_stream, _stream};

  T *grad_residual_ptr = buffer;
  buffer += _batch_dim;

  T *grad_ff1_inp_ptr = buffer;
  buffer += _batch_dim;

  T *dequant_ptr = buffer;
  buffer += _batch_tokens * _intermediate_size;

  T *grad_ff1_out_ptr = buffer;

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

  if (_enable_quant) {
    _ff2_inp_ptr = dequant_ptr;

    launch_dequantize<T>(_ff2_inp_ptr, _ff2_inp_i8_ptr, _output_cmax_ptr,
                         _batch_tokens * _intermediate_size, 2, _stream);
    _ff2.Backward(_batch_tokens, grad_inp_ptr, _ff2_inp_ptr, _output_w_ptr,
                  _grad_output_w_ptr, _grad_output_b_ptr, _cublasHandle,
                  _stream, grad_ff1_out_ptr, nullptr, false);

    // launch_d_cmax(_grad_output_w_ptr, _grad_output_cmax_ptr + 1,
    //               _ffn_activation_dropout.get_mask(),
    //               _intermediate_size * _hidden_size, 4, _stream);
    _ffn_activation_dropout.d_quant_bias_act_dropout(
        grad_ff1_out_ptr, _grad_inter_b_ptr, _grad_inter_cmax_ptr + 2,
        _grad_output_cmax_ptr, _act_inp_i8_ptr,
        _ffn_activation_dropout.get_mask(), _inter_cmax_ptr + 2,
        _ffn_activation_dropout.get_mask(), _inter_b_ptr, _batch_tokens,
        _intermediate_size, _activation_fn, _stream);

    launch_dequantize<T>(_ff1_inp_ptr, _ff1_inp_i8_ptr, _output_cmax_ptr,
                         _batch_tokens * _hidden_size, 2, _stream);

    _ff1.Backward(_batch_tokens, grad_ff1_out_ptr, _ff1_inp_ptr, _inter_w_ptr,
                  _grad_inter_w_ptr, _grad_inter_b_ptr, _cublasHandle, _stream,
                  grad_ff1_inp_ptr, nullptr, false);

    // launch_d_cmax(_grad_inter_w_ptr, _grad_inter_cmax_ptr + 1,
    //               _ffn_dropout.get_mask(), _intermediate_size * _hidden_size,
    //               4, _stream);

    if (_pre_or_postLayerNorm) {
      _ffn_ln.Backward(_grad_ffn_nw_ptr, _grad_ffn_nb_ptr, grad_inp_ptr,
                       _grad_inter_cmax_ptr, grad_ff1_inp_ptr, grad_output_ptr,
                       _ff1_inp_ptr, _ffn_nw_ptr, _ffn_nb_ptr,
                       _ffn_dropout.get_mask(), _batch_tokens, streams);

    } else {
      launch_d_cmax(grad_inp_ptr, _grad_inter_cmax_ptr, _ffn_dropout.get_mask(),
                    _batch_dim, 2, _stream);
      launch_fused_add2<T>(grad_inp_ptr, grad_ff1_inp_ptr, grad_residual_ptr,
                           _batch_size, _seq_len, _hidden_size, _stream);
    }

    return;
  }

  _ff2.Backward(_batch_tokens, grad_inp_ptr, _ff2_inp_ptr, _output_w_ptr,
                _grad_output_w_ptr, _grad_output_b_ptr, _cublasHandle, _stream,
                grad_ff1_out_ptr, nullptr, false);

  _ffn_activation_dropout.d_bias_act_dropout(
      grad_ff1_out_ptr, _grad_inter_b_ptr, _relu_inp_ptr, _inter_b_ptr,
      _batch_tokens, _intermediate_size, _activation_fn, _stream);

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
                         _batch_size, _seq_len, _hidden_size, _stream);
  }
}

template <typename T>
void TransformerEncoderLayer<T>::Backward(const T *grad_output_ptr,
                                          const T *input_ptr,
                                          const T *output_ptr,
                                          const T *input_mask_ptr,
                                          T *grad_input_ptr) {
  _stream = Context::Instance().get_stream();
  _cublasHandle = Context::Instance().get_cublashandle();
  T *grad_ffn_inp_ptr = _shared_mem_ptr;
  T *buffer = grad_ffn_inp_ptr + _batch_dim;

  /*
  buffer size needed by ffn bw:
      2 * _batch_dim + _batch_size * _seq_len * _intermediate_size
  */
  ffn_layer_bw(grad_output_ptr, output_ptr, grad_ffn_inp_ptr, buffer);

  /*
  buffer size needed by attn bw:
      4 * _batch_dim + max(3 * _batch_dim,
      _batch_size * _head_num * _seq_len * _seq_len);
  */
  attn_layer_bw(input_ptr, input_mask_ptr, grad_ffn_inp_ptr, grad_input_ptr,
                buffer);
}

template <typename T>
void TransformerEncoderLayer<T>::SetTrainingMode(bool training) {
  // Dropout will be skipped when not in training model.
  _attn_prob_dropout.SetTrainingMode(training);
  _attn_dropout.SetTrainingMode(training);
  _ffn_activation_dropout.SetTrainingMode(training);
  _ffn_dropout.SetTrainingMode(training);
}

template <typename T>
void TransformerEncoderLayer<T>::SetQuantMode(bool enable_quant) {
  if (_enable_quant != enable_quant) {
    free_independent_mem_buffer();
    _enable_quant = enable_quant;
    allocate_independent_mem_buffer();
  }
}

template <typename T>
T *TransformerEncoderLayer<T>::_shared_mem_ptr = nullptr;

template class TransformerEncoderLayer<float>;
template class TransformerEncoderLayer<__half>;
