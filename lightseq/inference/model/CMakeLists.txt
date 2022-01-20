cmake_minimum_required(VERSION 3.18)
find_package(CUDAToolkit)

set(transformer_files decoder.cc.cu encoder.cc.cu cublas_helper.cc)
add_library(transformer_model STATIC ${transformer_files})
target_link_libraries(transformer_model PUBLIC cuda_kernels)
target_link_libraries(transformer_model PUBLIC transformer_weight)
if(DYNAMIC_API)
  target_link_libraries(transformer_model PRIVATE CUDA::cublas CUDA::cublasLt)
else()
  target_link_libraries(transformer_model PRIVATE CUDA::cublas_static
                                                  CUDA::cublasLt_static)
endif()

target_include_directories(transformer_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(gpt_model STATIC gpt_encoder.cc.cu)
target_link_libraries(gpt_model PUBLIC cuda_kernels)
target_link_libraries(gpt_model PUBLIC gpt_weight)
if(DYNAMIC_API)
  target_link_libraries(gpt_model PRIVATE CUDA::cublas CUDA::cublasLt)
else()
  target_link_libraries(gpt_model PRIVATE CUDA::cublas_static
                                          CUDA::cublasLt_static)
endif()

target_include_directories(gpt_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(bert_model STATIC bert_encoder.cc.cu)
target_link_libraries(bert_model PUBLIC cuda_kernels)
target_link_libraries(bert_model PUBLIC bert_weight)
if(DYNAMIC_API)
  target_link_libraries(bert_model PRIVATE CUDA::cublas CUDA::cublasLt)
else()
  target_link_libraries(bert_model PRIVATE CUDA::cublas_static
                                           CUDA::cublasLt_static)
endif()
