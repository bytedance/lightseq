cmake_minimum_required(VERSION 3.18)

# (default) use C API for HDF5 library
find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

protobuf_generate_cpp(BERT_PROTO_SRC BERT_PROTO_HEADER bert.proto)

add_library(bert_weight STATIC bert_weight.cc ${BERT_PROTO_SRC}
                               ${BERT_PROTO_HEADER})
target_link_libraries(bert_weight PUBLIC utils ${Protobuf_LIBRARIES})
target_include_directories(bert_weight PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(bert_weight PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
