find_package(Threads REQUIRED)
find_package(CUDAToolkit)

add_library(core STATIC context.cpp node.cpp manager.cpp layer.cpp tensor.cpp)

target_link_libraries(core PUBLIC cuda_kernels)

if(DYNAMIC_API)
  target_link_libraries(core PRIVATE CUDA::cublas CUDA::cublasLt)
else()
  target_link_libraries(core PRIVATE CUDA::cublas_static CUDA::cublasLt_static)
endif()
