cmake_minimum_required(VERSION 3.18)

pybind11_add_module(
    lightseq MODULE 
    pybind_model.cpp
)

target_link_libraries(lightseq PUBLIC liblightseq)
target_link_libraries(lightseq PUBLIC utils)
set_target_properties(lightseq PROPERTIES OUTPUT_NAME inference)
