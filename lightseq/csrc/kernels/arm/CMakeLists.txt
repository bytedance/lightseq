cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

cmake_minimum_required(VERSION 3.18)
set(lightseq_kernel_files gemm.cc utils.cc)

add_library(lightseq_kernels STATIC ${lightseq_kernel_files})
target_include_directories(lightseq_kernels INTERFACE includes)
