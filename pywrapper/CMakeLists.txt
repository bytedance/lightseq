cmake_minimum_required(VERSION 3.18)

pybind11_add_module(pyseqlib MODULE transformer_decoder.cc.cu)
target_link_libraries(pyseqlib PUBLIC transformer_model)
