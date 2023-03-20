cmake_minimum_required(VERSION 3.18)


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


add_library(lightseq_kernels STATIC ${cuda_kernel_files})
target_link_libraries(lightseq_kernels PUBLIC -lcublas)
