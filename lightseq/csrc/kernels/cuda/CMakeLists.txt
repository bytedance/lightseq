find_package(CUDAToolkit)
cmake_minimum_required(VERSION 3.18)

find_package(CUDA 11 REQUIRED)

set(cuda_kernel_files
    util.cc.cu
    cross_entropy.cu
    cublas_wrappers.cu
    cuda_util.cu
    dropout_kernels.cu
    embedding_kernels.cu
    embKernels.cc.cu
    # fused_adam_kernel.cu
    general_kernels.cu
    gptKernels.cc.cu
    normalize_kernels.cu
    softmax_kernels.cu
    softmax_kernels_new.cu
    transform_kernels.cu
    transform_kernels_new.cu
    crf.cu
    transformerKernels.cc.cu)

list(APPEND CMAKE_MODULE_PATH ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

add_library(lightseq_kernels STATIC ${cuda_kernel_files})
target_link_libraries(lightseq_kernels PUBLIC -lcublas)
target_include_directories(lightseq_kernels PUBLIC ${HDF5_INCLUDE_DIRS})
target_include_directories(lightseq_kernels INTERFACE includes)
target_link_libraries(lightseq_kernels PRIVATE ${HDF5_LIBRARIES})
