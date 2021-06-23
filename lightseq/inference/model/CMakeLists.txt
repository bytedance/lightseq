cmake_minimum_required(VERSION 3.18)
find_package(CUDAToolkit)

set(transformer_files decoder.cc.cu encoder.cc.cu)
add_library(transformer_model STATIC ${transformer_files})
target_link_libraries(transformer_model PUBLIC cuda_kernels)
target_link_libraries(transformer_model PUBLIC transformer_weight)
target_link_libraries(transformer_model PUBLIC CUDA::cublas_static CUDA::cublasLt_static)
set_target_properties(transformer_model PROPERTIES CUDA_SEPARABLE_COMPILATION
                                                   ON)

add_library(gpt_model STATIC gpt_encoder.cc.cu)
target_link_libraries(gpt_model PUBLIC cuda_kernels)
target_link_libraries(gpt_model PUBLIC gpt_weight)
target_link_libraries(gpt_model PUBLIC CUDA::cublas_static CUDA::cublasLt_static)
set_target_properties(gpt_model PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
