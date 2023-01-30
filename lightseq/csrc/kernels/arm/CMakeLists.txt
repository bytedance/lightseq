cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

cmake_minimum_required(VERSION 3.18)
set(lightseq_kernel_files gemm.cc)

add_library(lightseq_kernels STATIC ${lightseq_kernel_files})
target_include_directories(lightseq_kernels PUBLIC ${HDF5_INCLUDE_DIRS})
target_include_directories(lightseq_kernels INTERFACE includes)
