#include "quant_linear_layer.h"

#include "context.h"
#include "kernels.h"

template <typename T>
QuantLinearLayer<T>::QuantLinearLayer(int layer_id, int in_features,
                                      int out_features, int max_batch_tokens)
    : _layer_id(layer_id),
      _in_features(in_features),
      _out_features(out_features),
      _max_batch_tokens(max_batch_tokens),
      _enable_quant(false),
      _linear(typename FeedForward<T>::Config(out_features, in_features)) {
  allocate_mem_buffer();
}

template <typename T>
QuantLinearLayer<T>::~QuantLinearLayer() {
  free_mem_buffer();
}

template <typename T>
void QuantLinearLayer<T>::Forward(const T *inputs_ptr, const T *weight_ptr,
                                  const T *cmax_ptr, T *outputs_ptr) {
  cudaStream_t stream = Context::Instance().get_stream();
  _cublasHandle = Context::Instance().get_cublashandle();
  _cublasLtHandle = Context::Instance().get_cublaslthandle();
  int tweaked_out_features = static_cast<int>((_out_features + 7) / 8) * 8;

  int batch_tokens = _batch_size * _seq_len;

  if (_enable_quant) {
    launch_quantize<T>(_quant_input_ptr, nullptr, _igemm_alpha_ptr, inputs_ptr,
                       cmax_ptr, _in_features * batch_tokens, 2, stream);

    launch_quantize<T>(_quant_weight_ptr, nullptr, nullptr, weight_ptr,
                       cmax_ptr + 1, _in_features * _out_features, 4, stream);

    _linear.Forward(batch_tokens, _quant_input_ptr, _quant_weight_ptr,
                    _igemm_alpha_ptr, _igemm_beta_ptr, _quant_output_ptr,
                    _cublasLtHandle, stream);

    launch_dequantize(outputs_ptr, _quant_output_ptr, cmax_ptr + 2,
                      batch_tokens * tweaked_out_features, 6, stream);
  } else {
    _linear.Forward(batch_tokens, inputs_ptr, weight_ptr, outputs_ptr,
                    _cublasHandle);
  }
}

template <typename T>
void QuantLinearLayer<T>::set_cur_batch_shape(int batch_size, int seq_len) {
  _batch_size = batch_size;
  _seq_len = seq_len;
}

template <typename T>
void QuantLinearLayer<T>::SetQuantMode(bool enable_quant) {
  if (_enable_quant != enable_quant) {
    _enable_quant = enable_quant;
    if (_enable_quant) {
      std::cout << "QuantLinearLayer #" << _layer_id << " enable quantization"
                << std::endl;
    }
  }
}

template class QuantLinearLayer<float>;
template class QuantLinearLayer<__half>;