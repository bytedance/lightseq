cmake_minimum_required(VERSION 3.18)

# (default) use C API for HDF5 library
find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})


set(PROTO_FILES bert.proto bert_crf.proto transformer.proto gpt.proto)

set(WEIGHT_FILES bert_weight.cc # bert_crf_weight.cc
                 transformer_weight.cc gpt_weight.cc)

protobuf_generate_cpp(PROTO_SRC PROTO_HEADER ${PROTO_FILES})
add_library(weight_lib STATIC ${WEIGHT_FILES} ${PROTO_SRC} ${PROTO_HEADER}
                              proto_util.cc)
target_link_libraries(weight_lib PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(weight_lib PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(weight_lib PUBLIC lightseq_kernels)

target_include_directories(weight_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(weight_lib PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(weight_lib PUBLIC ${HDF5_INCLUDE_DIRS})
