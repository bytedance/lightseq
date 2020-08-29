cmake_minimum_required(VERSION 3.18)
find_package(Protobuf REQUIRED)

pybind11_add_module(pyseqlib MODULE transformer_decoder.cu.cc)
target_link_libraries(pyseqlib PUBLIC -lcublas -lcudart -lprotobuf)
target_link_libraries(pyseqlib PUBLIC transformer_model)
target_link_libraries(pyseqlib PUBLIC ${Protobuf_LIBRARY})
set_target_properties(pyseqlib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
