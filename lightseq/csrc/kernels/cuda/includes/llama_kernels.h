#pragma once 
#include <cuda.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include "kernels.h"
#include <stdexcept>

namespace lightseq {
namespace cuda {

template <typename T>
void launch_rotary_position_qk(const T *input_ptr, const T *sin_ptr,
                               const T *cos_ptr, T *output_ptr, size_t max_step,
                               size_t batch_size, size_t nhead,
                               size_t offset_seq_len, size_t query_len,
                               size_t head_dim, cudaStream_t stream);


template <typename T>
void launch_elewise_product_silu(const T* inpA_ptr, const T* inpB_ptr, T* out_ptr, size_t batch_size, size_t seq_len, size_t inner_size, cudaStream_t stream);

}
}