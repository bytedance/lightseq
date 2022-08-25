find_package(CUDAToolkit)

set(cuda_kernel_files
    cross_entropy.cu
    cublas_wrappers.cu
    cuda_util.cu
    dropout_kernels.cu
    embedding_kernels.cu
    embKernels.cc.cu
    # fused_adam_kernel.cu
    general_kernels.cu
    normalize_kernels.cu
    softmax_kernels.cu
    softmax_kernels_new.cu
    transform_kernels.cu
    transform_kernels_new.cu)

add_library(cuda_kernels STATIC ${cuda_kernel_files})
target_include_directories(cuda_kernels INTERFACE includes)
