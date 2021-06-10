#include "cross_entropy_layer.h"

#include "context.h"
#include "kernels.h"

template <typename T>
CrossEntropyLayer<T>::CrossEntropyLayer(float epsilon, int padding_idx,
                                        int max_batch_tokens)
    : _epsilon(epsilon),
      _padding_idx(padding_idx),
      _max_batch_tokens(max_batch_tokens) {
  allocate_mem_buffer();
}

template <typename T>
CrossEntropyLayer<T>::~CrossEntropyLayer() {
  free_mem_buffer();
}

template <typename T>
void CrossEntropyLayer<T>::Forward(const T *inputs_ptr, const int *targets_ptr,
                                   float *outputs_ptr, float *nll_loss_ptr) {
  cudaStream_t stream = Context::Instance().get_stream();

  launch_cross_entropy_fw<T>(inputs_ptr, targets_ptr, outputs_ptr, nll_loss_ptr,
                             _loss_buffer, _padding_idx, _epsilon, _batch_size,
                             _seq_len, _vocab_size, stream);
}

template <typename T>
void CrossEntropyLayer<T>::Backward(const float *grad_outputs_ptr,
                                    const T *inputs_ptr, const int *targets_ptr,
                                    T *grad_inputs_ptr) {
  cudaStream_t stream = Context::Instance().get_stream();

  launch_cross_entropy_bw<T>(grad_outputs_ptr, inputs_ptr, targets_ptr,
                             grad_inputs_ptr, _padding_idx, _epsilon,
                             _batch_size, _seq_len, _vocab_size, stream);
}

template <typename T>
void CrossEntropyLayer<T>::set_cur_batch_shape(int batch_size, int seq_len,
                                               int vocab_size) {
  _batch_size = batch_size;
  _seq_len = seq_len;
  _vocab_size = vocab_size;
}

template class CrossEntropyLayer<float>;
template class CrossEntropyLayer<__half>;
