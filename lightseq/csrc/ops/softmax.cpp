#include"softmax.h"

template <typename T>
void Softmax<T>::Forward(T *vals, const T *attn_mask, int batch_size, int from_len,
               int to_len, cudaStream_t &stream, bool mask_future) {
  launch_attn_softmax<T>(vals, attn_mask, batch_size, config_.nhead, from_len,
    to_len, config_.mask_future | mask_future, stream);
}

template <typename T>
void Softmax<T>::Backward(T *out_grad, const T *soft_out, int batch_size, int from_len,
                int to_len, cudaStream_t stream) {
    launch_attn_softmax_bw<T>(out_grad, soft_out,
        batch_size * config_.nhead * from_len, to_len,
        stream);
}

template class Softmax<float>;
template class Softmax<__half>;