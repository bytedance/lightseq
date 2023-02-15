cmake_minimum_required(VERSION 3.18)

if(DEVICE_INDEX EQUAL 1)
  set(LS_PYBIND_FILES pybind_kernel_x86.cpp)
elseif(DEVICE_INDEX EQUAL 0)
  set(LS_PYBIND_FILES pybind_kernel.cpp)
endif()

pybind11_add_module(lightseq MODULE ${LS_PYBIND_FILES})
target_link_libraries(lightseq PUBLIC liblightseq lightseq_kernels)
set_target_properties(lightseq PROPERTIES OUTPUT_NAME inference)
