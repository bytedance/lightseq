
find_package(CUDAToolkit)

set(cuda_kernel_files
    cuda_util.cu
    transform_kernels.cu
    embedding_kernels.cu
)

add_library(cuda_kernels STATIC ${cuda_kernel_files})
target_include_directories(cuda_kernels INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})