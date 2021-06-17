#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <type_traits>

#include "cuda_util.h"
#include "dropout.h"
#include "feed_forward.h"
#include "normalize_layer.h"
#include "softmax.h"
#include "strided_batch_gemm.h"

template <typename T>
class TransformerDecoderLayer {
 public:
  TransformerDecoderLayer(int layer_id, int max_batch_tokens, int _max_seq_len,
                          int hidden_size, int num_heads, int intermediate_size,
                          float attn_dropout_ratio,
                          float hidden_output_dropout_ratio,
                          float layer_norm_eps, bool pre_or_postLayerNorm);

  virtual ~TransformerDecoderLayer();

  void Forward(const T *dec_input_ptr, const T *enc_output_ptr,
               const T *enc_mask_ptr, T *dec_output_ptr,
               std::vector<T *> &cache);

  void Backward(const T *grad_dec_output_ptr, const T *dec_input_ptr,
                const T *enc_output_ptr, const T *enc_mask_ptr,
                const T *dec_output_ptr, T *grad_dec_input_ptr,
                T *grad_enc_output_ptr);

  void encdec_kv_fw(const T *enc_output_ptr);

  void self_attn_layer_fw(const T *input_ptr, T *output_ptr, T *buffer,
                          std::vector<T *> &cache);

  void encdec_attn_layer_fw(const T *input_ptr, const T *enc_mask_ptr,
                            T *output_ptr, T *buffer);

  void ffn_layer_fw(T *inp_ptr, T *out_ptr);

  void encdec_kv_bw(const T *enc_output_ptr, T *grad_enc_output_ptr);

  void self_attn_layer_bw(const T *input_ptr, const T *output_ptr,
                          const T *grad_output_ptr, T *grad_input_ptr,
                          T *buffer);

  void encdec_attn_layer_bw(const T *output_ptr, const T *grad_output_ptr,
                            T *grad_input_ptr, T *buffer);

  void ffn_layer_bw(const T *grad_output_ptr, const T *output_ptr,
                    T *grad_inp_ptr, T *buffer);

  void set_cur_batch_shape(int batch_size, int trg_seq_len, int src_seq_len,
                           int step = -1) {
    _batch_size = batch_size;
    _trg_seq_len = trg_seq_len;  // beam_size for inference
    _src_seq_len = src_seq_len;
    _batch_tokens = batch_size * trg_seq_len;
    _batch_heads = batch_size * _heads;
    _batch_dim = _batch_tokens * _hidden_size;

    _encdec_attn_scores.SetConfig(_src_seq_len, _trg_seq_len,
                                  _hidden_size / _heads);
    _encdec_attn_context.SetConfig(_hidden_size / _heads, _trg_seq_len,
                                   _src_seq_len);
    if (step >= 0) {
      _predict = true;
      _step = step;
      _attn_scores.SetConfig(step + 1, 1, _hidden_size / _heads);
      _attn_context.SetConfig(_hidden_size / _heads, 1, step + 1);
    } else {
      _predict = false;
      _step = -1;
      _attn_scores.SetConfig(_trg_seq_len, _trg_seq_len, _hidden_size / _heads);
      _attn_context.SetConfig(_hidden_size / _heads, _trg_seq_len,
                              _trg_seq_len);
    }
  }

  void SetTrainingMode(bool training) {
    _training = training;
    _attn_prob_dropout.SetTrainingMode(training);
    _attn_dropout.SetTrainingMode(training);
    _encdec_attn_prob_dropout.SetTrainingMode(training);
    _encdec_attn_dropout.SetTrainingMode(training);
    _ffn_activation_dropout.SetTrainingMode(training);
    _ffn_dropout.SetTrainingMode(training);
  }

  void assign_weight_ptr(const T *weights_ptr) {
    const T *wptr = weights_ptr;
    // assign weights ptr
    _attn_qkvw_ptr = wptr;
    wptr += _hidden_size * _hidden_size * 3;
    _attn_qkvb_ptr = wptr;
    wptr += _hidden_size * 3;
    _attn_ow_ptr = wptr;
    wptr += _hidden_size * _hidden_size;
    _attn_ob_ptr = wptr;
    wptr += _hidden_size;
    _attn_nw_ptr = wptr;
    wptr += _hidden_size;
    _attn_nb_ptr = wptr;
    wptr += _hidden_size;

    _encdec_attn_qw_ptr = wptr;
    wptr += _hidden_size * _hidden_size;
    _encdec_attn_qb_ptr = wptr;
    wptr += _hidden_size;
    _encdec_attn_ow_ptr = wptr;
    wptr += _hidden_size * _hidden_size;
    _encdec_attn_ob_ptr = wptr;
    wptr += _hidden_size;
    _encdec_attn_nw_ptr = wptr;
    wptr += _hidden_size;
    _encdec_attn_nb_ptr = wptr;
    wptr += _hidden_size;

    _inter_w_ptr = wptr;
    wptr += _hidden_size * _intermediate_size;
    _inter_b_ptr = wptr;
    wptr += _intermediate_size;
    _output_w_ptr = wptr;
    wptr += _hidden_size * _intermediate_size;
    _output_b_ptr = wptr;
    wptr += _hidden_size;
    _ffn_nw_ptr = wptr;
    wptr += _hidden_size;
    _ffn_nb_ptr = wptr;
    wptr += _hidden_size;

    if (_layer_id == 0) {
      _encdec_attn_kvw_ptr = wptr;
      wptr += _shared_nlayer * _hidden_size * _hidden_size * 2;
      _encdec_attn_kvb_ptr = wptr;
      wptr += _shared_nlayer * _hidden_size * 2;
    } else {
      _encdec_attn_kvw_ptr = nullptr;
      _encdec_attn_kvb_ptr = nullptr;
    }
  }

  void assign_grad_ptr(T *grads_ptr) {
    T *gptr = grads_ptr;
    // assign weights ptr
    _grad_attn_qkvw_ptr = gptr;
    gptr += _hidden_size * _hidden_size * 3;
    _grad_attn_qkvb_ptr = gptr;
    gptr += _hidden_size * 3;
    _grad_attn_ow_ptr = gptr;
    gptr += _hidden_size * _hidden_size;
    _grad_attn_ob_ptr = gptr;
    gptr += _hidden_size;
    _grad_attn_nw_ptr = gptr;
    gptr += _hidden_size;
    _grad_attn_nb_ptr = gptr;
    gptr += _hidden_size;

    _grad_encdec_attn_qw_ptr = gptr;
    gptr += _hidden_size * _hidden_size;
    _grad_encdec_attn_qb_ptr = gptr;
    gptr += _hidden_size;
    _grad_encdec_attn_ow_ptr = gptr;
    gptr += _hidden_size * _hidden_size;
    _grad_encdec_attn_ob_ptr = gptr;
    gptr += _hidden_size;
    _grad_encdec_attn_nw_ptr = gptr;
    gptr += _hidden_size;
    _grad_encdec_attn_nb_ptr = gptr;
    gptr += _hidden_size;

    _grad_inter_w_ptr = gptr;
    gptr += _hidden_size * _intermediate_size;
    _grad_inter_b_ptr = gptr;
    gptr += _intermediate_size;
    _grad_output_w_ptr = gptr;
    gptr += _hidden_size * _intermediate_size;
    _grad_output_b_ptr = gptr;
    gptr += _hidden_size;
    _grad_ffn_nw_ptr = gptr;
    gptr += _hidden_size;
    _grad_ffn_nb_ptr = gptr;
    gptr += _hidden_size;

    if (_layer_id == 0) {
      _grad_encdec_attn_kvw_ptr = gptr;
      gptr += _shared_nlayer * _hidden_size * _hidden_size * 2;
      _grad_encdec_attn_kvb_ptr = gptr;
      gptr += _shared_nlayer * _hidden_size * 2;
    } else {
      _grad_encdec_attn_kvw_ptr = nullptr;
      _grad_encdec_attn_kvb_ptr = nullptr;
    }
  }

 private:
  void allocate_buffer() {
    // allocate local gpu memory
    if (_pre_or_postLayerNorm) {
      _gemmQKV_inp_ptr = cuda_malloc<T>(_max_batch_tokens * _hidden_size);
    } else {
      _gemmQKV_inp_ptr = nullptr;
    }
    _qkv_ptr = cuda_malloc<T>(_max_batch_tokens * _hidden_size * 3);
    _soft_out_ptr = cuda_malloc<T>(_max_batch_tokens * _heads * _max_seq_len);
    _attn_score_ptr = cuda_malloc<T>(_max_batch_tokens * _heads * _max_seq_len);
    _attn_output_ptr = cuda_malloc<T>(_max_batch_tokens * _hidden_size);

    _gemmQ_inp_ptr = cuda_malloc<T>(_max_batch_tokens * _hidden_size);
    _encdec_q_ptr = cuda_malloc<T>(_max_batch_tokens * _hidden_size);
    _encdec_soft_out_ptr =
        cuda_malloc<T>(_max_batch_tokens * _heads * _max_seq_len);
    _encdec_attn_score_ptr =
        cuda_malloc<T>(_max_batch_tokens * _heads * _max_seq_len);
    _encdec_attn_output_ptr = cuda_malloc<T>(_max_batch_tokens * _hidden_size);

    _ff1_inp_ptr = cuda_malloc<T>(_max_batch_tokens * _hidden_size);
    _relu_inp_ptr = cuda_malloc<T>(_max_batch_tokens * _intermediate_size);
    _ff2_inp_ptr = cuda_malloc<T>(_max_batch_tokens * _intermediate_size);

    // buffer size needed by ffn bw
    size_t sz_ffn_bw = 3 * _max_batch_tokens * _hidden_size +
                       _max_batch_tokens * _intermediate_size;
    // buffer size needed by attn bw
    size_t sz_attn_bw = 5 * _max_batch_tokens * _hidden_size +
                        std::max(3 * _max_batch_tokens * _hidden_size,
                                 _max_batch_tokens * _heads * _max_seq_len);
    size_t smem_size = std::max(sz_ffn_bw, sz_attn_bw);

    if (!_shared_buffer_ptr) {
      _shared_buffer_ptr = cuda_malloc<T>(smem_size);
      std::cout << "Decoder layer #" << _layer_id
                << " allocate shared memory size: " << smem_size << std::endl;
    }
  }

  void allocate_encdec_kv_memory() {
    // should be run after all decoder layers have been initialized.
    if (_shared_encdec_kv_ptr) {
      return;
    }
    size_t smem_size = _shared_nlayer * _max_batch_tokens * _hidden_size * 2;
    _shared_encdec_kv_ptr = cuda_malloc<T>(smem_size);
    _shared_grad_encdec_kv_ptr = cuda_malloc<T>(smem_size);
    _encdec_kv_linear.reset_size(_shared_nlayer * 2 * _hidden_size,
                                 _hidden_size);
    std::cout << "Decoder layer #" << _layer_id << " allocate encdec_kv memory"
              << std::endl;
  }

  void free_memory() {
    // free local gpu memory
    cuda_free(_gemmQKV_inp_ptr);
    cuda_free(_qkv_ptr);
    cuda_free(_soft_out_ptr);
    cuda_free(_attn_score_ptr);
    cuda_free(_attn_output_ptr);

    cuda_free(_gemmQ_inp_ptr);
    cuda_free(_encdec_q_ptr);
    cuda_free(_encdec_soft_out_ptr);
    cuda_free(_encdec_attn_score_ptr);
    cuda_free(_encdec_attn_output_ptr);

    cuda_free(_ff1_inp_ptr);
    cuda_free(_relu_inp_ptr);
    cuda_free(_ff2_inp_ptr);

    // free shared gpu memory between layers
    cuda_free(_shared_buffer_ptr);
    _shared_buffer_ptr = nullptr;
    cuda_free(_shared_encdec_kv_ptr);
    _shared_encdec_kv_ptr = nullptr;
    cuda_free(_shared_grad_encdec_kv_ptr);
    _shared_grad_encdec_kv_ptr = nullptr;
  }

  // const parameter between batch
  const size_t _layer_id;
  const size_t _hidden_size;
  const size_t _heads;
  const size_t _intermediate_size;
  const size_t _max_batch_tokens;
  const size_t _max_seq_len;
  const bool _pre_or_postLayerNorm;
  // dynamic parameter between batch
  size_t _batch_size;
  size_t _trg_seq_len;
  size_t _src_seq_len;
  size_t _batch_tokens;
  size_t _batch_heads;
  size_t _batch_dim;
  int _step;
  bool _training;
  bool _predict;

  cublasHandle_t _cublasHandle;
  cudaStream_t _stream;

  // layers
  Normalize_Layer<T> _attn_ln, _encdec_attn_ln, _ffn_ln;
  FeedForward<T> _qkv_linear, _attn_out_linear, _encdec_q_linear,
      _encdec_kv_linear, _encdec_attn_out_linear, _ff1, _ff2;
  Softmax<T> _softmax, _encdec_softmax;
  Dropout<T> _attn_prob_dropout, _attn_dropout, _encdec_attn_prob_dropout,
      _encdec_attn_dropout, _ffn_activation_dropout, _ffn_dropout;
  StridedBatchGemm<T> _attn_scores, _attn_context, _encdec_attn_scores,
      _encdec_attn_context;

  // layer's local GPU memory
  T *_gemmQKV_inp_ptr;
  T *_qkv_ptr;
  T *_soft_out_ptr;
  T *_attn_score_ptr;
  T *_attn_output_ptr;

  T *_gemmQ_inp_ptr;
  T *_encdec_q_ptr;
  T *_encdec_soft_out_ptr;
  T *_encdec_attn_score_ptr;
  T *_encdec_attn_output_ptr;

  T *_ff1_inp_ptr;
  T *_relu_inp_ptr;
  T *_ff2_inp_ptr;

  // shared GPU memory between layer
  static size_t _shared_nlayer;
  static T *_shared_buffer_ptr;
  static T *_shared_encdec_kv_ptr;
  static T *_shared_grad_encdec_kv_ptr;
  static T *_shared_infer_encdec_kv_ptr;

  // weights ptr
  const T *_attn_qkvw_ptr;
  const T *_attn_qkvb_ptr;
  const T *_attn_ow_ptr;
  const T *_attn_ob_ptr;
  const T *_attn_nw_ptr;
  const T *_attn_nb_ptr;

  const T *_encdec_attn_qw_ptr;
  const T *_encdec_attn_qb_ptr;
  const T *_encdec_attn_ow_ptr;
  const T *_encdec_attn_ob_ptr;
  const T *_encdec_attn_nw_ptr;
  const T *_encdec_attn_nb_ptr;
  const T *_encdec_attn_kvw_ptr;
  const T *_encdec_attn_kvb_ptr;

  const T *_inter_w_ptr;
  const T *_inter_b_ptr;
  const T *_output_w_ptr;
  const T *_output_b_ptr;
  const T *_ffn_nw_ptr;
  const T *_ffn_nb_ptr;

  // grads ptr
  T *_grad_attn_qkvw_ptr;
  T *_grad_attn_qkvb_ptr;
  T *_grad_attn_ow_ptr;
  T *_grad_attn_ob_ptr;
  T *_grad_attn_nw_ptr;
  T *_grad_attn_nb_ptr;

  T *_grad_encdec_attn_qw_ptr;
  T *_grad_encdec_attn_qb_ptr;
  T *_grad_encdec_attn_kvw_ptr;
  T *_grad_encdec_attn_kvb_ptr;
  T *_grad_encdec_attn_ow_ptr;
  T *_grad_encdec_attn_ob_ptr;
  T *_grad_encdec_attn_nw_ptr;
  T *_grad_encdec_attn_nb_ptr;

  T *_grad_inter_w_ptr;
  T *_grad_inter_b_ptr;
  T *_grad_output_w_ptr;
  T *_grad_output_b_ptr;
  T *_grad_ffn_nw_ptr;
  T *_grad_ffn_nb_ptr;
};
