cmake_minimum_required(VERSION 3.18)

pybind11_add_module(lightseq MODULE pybind_kernel_x86.cpp)
target_link_libraries(lightseq PUBLIC liblightseq lightseq_kernels)
set_target_properties(lightseq PROPERTIES OUTPUT_NAME kernel)
