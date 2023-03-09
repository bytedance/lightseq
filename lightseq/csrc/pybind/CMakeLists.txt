cmake_minimum_required(VERSION 3.18)

set(LS_PYBIND_KERNEL_FILES pybind_model.cpp)
pybind11_add_module(lightseq MODULE ${LS_PYBIND_KERNEL_FILES})
target_link_libraries(lightseq PUBLIC liblightseq lightseq_kernels)
set_target_properties(lightseq PROPERTIES OUTPUT_NAME inference)
