cmake_minimum_required(VERSION 3.18)
find_package(CUDAToolkit)

set(transformer_files decoder.cc.cu encoder.cc.cu)
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

set(t5_files t5_decoder.cc.cu t5_encoder.cc.cu)
add_library(t5_model STATIC ${t5_files})
target_link_libraries(t5_model PUBLIC cuda_kernels)
target_link_libraries(t5_model PUBLIC t5_weight)
if(DYNAMIC_API)
  target_link_libraries(t5_model PRIVATE CUDA::cublas CUDA::cublasLt)
else()
  target_link_libraries(t5_model PRIVATE CUDA::cublas_static
                                         CUDA::cublasLt_static)
endif()

target_include_directories(t5_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


set(mt5_files mt5_decoder.cc.cu mt5_encoder.cc.cu)
add_library(mt5_model STATIC ${mt5_files})
target_link_libraries(mt5_model PUBLIC cuda_kernels)
target_link_libraries(mt5_model PUBLIC mt5_weight)
if(DYNAMIC_API)
  target_link_libraries(mt5_model PRIVATE CUDA::cublas CUDA::cublasLt)
else()
  target_link_libraries(mt5_model PRIVATE CUDA::cublas_static
                                         CUDA::cublasLt_static)
endif()

target_include_directories(mt5_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


set(quant_transformer_files quant_decoder.cc.cu quant_encoder.cc.cu
                            cublas_helper.cc)
add_library(quant_transformer_model STATIC ${quant_transformer_files})
target_link_libraries(quant_transformer_model PUBLIC cuda_kernels)
target_link_libraries(quant_transformer_model PUBLIC quant_transformer_weight)
if(DYNAMIC_API)
  target_link_libraries(quant_transformer_model PRIVATE CUDA::cublas
                                                        CUDA::cublasLt)
else()
  target_link_libraries(quant_transformer_model PRIVATE CUDA::cublas_static
                                                        CUDA::cublasLt_static)
endif()

target_include_directories(quant_transformer_model
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

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

add_library(quant_gpt_model STATIC quant_gpt_encoder.cc.cu)
target_link_libraries(quant_gpt_model PUBLIC cuda_kernels)
target_link_libraries(quant_gpt_model PUBLIC quant_gpt_weight)
if(DYNAMIC_API)
  target_link_libraries(quant_gpt_model PRIVATE CUDA::cublas CUDA::cublasLt)
else()
  target_link_libraries(quant_gpt_model PRIVATE CUDA::cublas_static
                                                CUDA::cublasLt_static)
endif()

target_include_directories(quant_gpt_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(bert_model STATIC bert_encoder.cc.cu)
target_link_libraries(bert_model PUBLIC cuda_kernels)
target_link_libraries(bert_model PUBLIC bert_weight)
if(DYNAMIC_API)
  target_link_libraries(bert_model PRIVATE CUDA::cublas CUDA::cublasLt)
else()
  target_link_libraries(bert_model PRIVATE CUDA::cublas_static
                                           CUDA::cublasLt_static)
endif()

add_library(quant_bert_model STATIC quant_bert_encoder.cc.cu)
target_link_libraries(quant_bert_model PUBLIC cuda_kernels)
target_link_libraries(quant_bert_model PUBLIC quant_bert_weight)
if(DYNAMIC_API)
  target_link_libraries(quant_bert_model PRIVATE CUDA::cublas CUDA::cublasLt)
else()
  target_link_libraries(quant_bert_model PRIVATE CUDA::cublas_static
                                                 CUDA::cublasLt_static)
endif()

set(moe_files moe_decoder.cc.cu moe_encoder.cc.cu)
add_library(moe_model STATIC ${moe_files})
target_link_libraries(moe_model PUBLIC cuda_kernels)
target_link_libraries(moe_model PUBLIC moe_weight)
if(DYNAMIC_API)
  target_link_libraries(moe_model PRIVATE CUDA::cublas CUDA::cublasLt)
else()
  target_link_libraries(moe_model PRIVATE CUDA::cublas_static
                                          CUDA::cublasLt_static)
endif()

add_library(vit_model STATIC vit_encoder.cc.cu)
target_link_libraries(vit_model PUBLIC cuda_kernels)
target_link_libraries(vit_model PUBLIC vit_weight)
if(DYNAMIC_API)
  target_link_libraries(vit_model PRIVATE CUDA::cublas CUDA::cublasLt)
else()
  target_link_libraries(vit_model PRIVATE CUDA::cublas_static
                                          CUDA::cublasLt_static)
endif()
