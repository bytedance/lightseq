#include "dropout.h"
#include <cuda.h>
#include <cuda_fp16.h>

template <typename T>
Dropout<T>::Dropout(const Dropout<T>::Config &config, size_t max_ele_num)
    : _config(config), _mask(nullptr) {
  _mask = cuda_malloc<uint8_t>(max_ele_num);
}

template <typename T>
Dropout<T>::~Dropout() { cuda_free(_mask); }

template <typename T>
void Dropout<T>::dropout(T *output, const T *input, int count, cudaStream_t stream,
            bool bwd) {
  launch_ls_dropout<T>(output, input, _mask, count, _config.RATIO(), stream,
                        bwd);
}

template <typename T>
void Dropout<T>::d_dropout(T *d_inp_out, int count, cudaStream_t stream) {
  launch_ls_dropout<T>(d_inp_out, d_inp_out, _mask, count, _config.RATIO(),
                        stream, true);
}

template <typename T>
void Dropout<T>::bias_dropout_residual(T *output, const T *input, const T *residual,
                            const T *bias, int rows, int cols,
                            cudaStream_t stream) {
  launch_ls_dropout_res_bias<T>(output, input, _mask, bias, residual,
                                rows * cols, cols, _config.RATIO(), stream);
}

template <typename T>
void Dropout<T>::d_bias_dropout_residual(T *d_input, T *d_bias, const T *d_output,
                            int rows, int cols, cudaStream_t stream) {
  launch_ls_dropout_bias_bwd<T>(d_input, d_bias, d_output, _mask, rows, cols,
                                _config.RATIO(), stream);
}

template <typename T>
void Dropout<T>::bias_act_dropout(T *output, const T *input, const T *bias, int rows,
                    int cols, std::string activation_fn,
                    cudaStream_t stream) {
  if (activation_fn == "relu") {
    launch_ls_dropout_act_bias<ActivationType::kRelu, T>(
        output, input, _mask, bias, rows * cols, cols, _config.RATIO(),stream);
  } else if (activation_fn == "gelu") {
    launch_ls_dropout_act_bias<ActivationType::kGelu, T>(output, input, _mask, bias, rows * cols, cols, _config.RATIO(), stream);
  } else {
    throw std::runtime_error("not supported activation: " + activation_fn);
  }
}

template <typename T>
void Dropout<T>::d_bias_act_dropout(T *d_inp_out, T *d_bias_out, const T *input,
                          const T *bias, int rows, int cols,
                          std::string activation_fn, cudaStream_t stream) {
  if (activation_fn == "relu") {
    launch_ls_dropout_act_bias_bwd<ActivationType::kRelu, T>(
        d_inp_out, d_bias_out, input, bias, d_inp_out, _mask, rows, cols,
        _config.RATIO(), stream);
  } else if (activation_fn == "gelu") {
    launch_ls_dropout_act_bias_bwd<ActivationType::kGelu, T>(
        d_inp_out, d_bias_out, input, bias, d_inp_out, _mask, rows, cols,
        _config.RATIO(), stream);
  } else {
    throw std::runtime_error("not supported activation: " + activation_fn);
  }
}

template <typename T>
bool Dropout<T>::HasDropout() const { return _config.RATIO() > 0.0; }

template <typename T>
void Dropout<T>::SetTrainingMode(bool training) { _config.training = training; }


template class Dropout<float>;
template class Dropout<__half>;