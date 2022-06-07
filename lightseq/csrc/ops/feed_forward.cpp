#include "feed_forward.h"

template <typename T>
void FeedForward<T>::Forward(int bsz, const T *input_ptr, const T *weights, T *out,
               cublasHandle_t &_cublasHandle) {
    float alpha = T(1.);
    float beta = T(0.);

    cublas_gemm_ex(_cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, config_.outputSize,
                    bsz, config_.inputSize, &alpha, &beta, weights, input_ptr,
                    out, cublasGemmAlgo_t(config_.gemm_algos[0]));
}

template <typename T>
void FeedForward<T>::Backward(int bsz, const T *out_grad, const T *input_ptr,
            const T *weights, T *weights_grad, T *bias_grad,
            cublasHandle_t &_cublasHandle, cudaStream_t &stream,
            T *inp_grad_out, T *out_grad_trans_out,
            bool compute_bias) {
    float alpha = (T)1.0, beta = (T)0.0;
    cublas_gemm_ex(_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, config_.inputSize,
                config_.outputSize, bsz, &alpha, &beta, input_ptr, out_grad,
                weights_grad, cublasGemmAlgo_t(config_.gemm_algos[1]));

    cublas_gemm_ex(_cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, config_.inputSize,
                bsz, config_.outputSize, &alpha, &beta, weights, out_grad,
                inp_grad_out, cublasGemmAlgo_t(config_.gemm_algos[2]));
    if (compute_bias) {
    launch_fuse_transpose_bias_kernel<T>(out_grad, bias_grad, bsz,
                                        config_.outputSize, stream);
    }
}
  
template <typename T>
void FeedForward<T>::reset_size(int outputSize, int inputSize) {
    config_.outputSize = outputSize;
    config_.inputSize = inputSize;
}

template class FeedForward<float>;
template class FeedForward<__half>;