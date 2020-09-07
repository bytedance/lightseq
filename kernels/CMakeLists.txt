cmake_minimum_required(VERSION 3.18)

set(cuda_kernel_files gptKernels.cc.cu transformerKernels.cc.cu)

add_library(cuda_kernels STATIC ${cuda_kernel_files})

target_include_directories(cuda_kernels INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
# target_link_libraries(cuda_kernels PUBLIC -lcublas -lcudart)
# set_target_properties(cuda_kernels PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(cuda_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
