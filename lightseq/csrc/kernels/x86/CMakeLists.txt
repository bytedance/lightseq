cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

cmake_minimum_required(VERSION 3.18)
# (default) use C API for HDF5 library
find_package(HDF5 REQUIRED)

set(lightseq_kernel_files util.cc gemm.cpp)

add_library(lightseq_kernels STATIC ${lightseq_kernel_files})
target_include_directories(lightseq_kernels PUBLIC ${HDF5_INCLUDE_DIRS})
target_include_directories(lightseq_kernels INTERFACE includes)
target_link_libraries(lightseq_kernels PRIVATE ${HDF5_LIBRARIES})
