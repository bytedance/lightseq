#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <type_traits>

#include "feed_forward.h"
#include "cublas_wrappers.h"
#include "cuda_util.h"

template <typename T>
class QuantLinearLayer {
 public:
  QuantLinearLayer(int layer_id, int in_features, int out_features,
                   int max_batch_tokens);

  virtual ~QuantLinearLayer();

  void Forward(const T *inputs_ptr, const T *weight_ptr, const T *cmax_ptr,
               T *outputs_ptr);

  void set_cur_batch_shape(int batch_size, int seq_len);

  void SetQuantMode(bool enable_quant);
  inline bool IsTrainingMode() const { return _training; }

 private:
  void allocate_mem_buffer() {
    int tweaked_out_features = static_cast<int>((_out_features + 7) / 8) * 8;

    // allocate local gpu memory
    _quant_input_ptr = cuda_malloc<int8_t>(_max_batch_tokens * _in_features);
    _quant_weight_ptr = cuda_malloc<int8_t>(_in_features * _out_features);
    _quant_output_ptr =
        cuda_malloc<int8_t>(_max_batch_tokens * tweaked_out_features);
    _igemm_alpha_ptr = cuda_malloc<float>(_max_batch_tokens);
    _igemm_beta_ptr = cuda_malloc<float>(1);
    cuda_set<float>(_igemm_beta_ptr, 0, 1);
  }

  void free_mem_buffer() {
    // free local gpu memory
    cuda_free(_quant_input_ptr);
    cuda_free(_quant_weight_ptr);
    cuda_free(_quant_output_ptr);
    cuda_free(_igemm_alpha_ptr);
    cuda_free(_igemm_beta_ptr);
  }
  FeedForward<T> _linear;

  cublasHandle_t _cublasHandle;
  cublasLtHandle_t _cublasLtHandle;

  const int _max_batch_tokens;
  const int _in_features;
  const int _out_features;

  bool _training;
  bool _enable_quant;

  const size_t _layer_id;
  size_t _batch_size;
  size_t _seq_len;

  int8_t *_quant_input_ptr;
  int8_t *_quant_weight_ptr;
  int8_t *_quant_output_ptr;
  float *_igemm_alpha_ptr;
  float *_igemm_beta_ptr;
};
