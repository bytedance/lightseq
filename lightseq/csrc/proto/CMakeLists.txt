cmake_minimum_required(VERSION 3.18)

# (default) use C API for HDF5 library
find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

protobuf_generate_cpp(BERT_PROTO_SRC BERT_PROTO_HEADER bert.proto)
protobuf_generate_cpp(QUANT_BERT_PROTO_SRC QUANT_BERT_PROTO_HEADER quant_bert.proto)
protobuf_generate_cpp(BERT_CRF_PROTO_SRC BERT_CRF_PROTO_HEADER bert_crf.proto)

add_library(bert_weight STATIC bert_weight.cc ${BERT_PROTO_SRC}
                               ${BERT_PROTO_HEADER})
target_link_libraries(bert_weight PUBLIC utils ${Protobuf_LIBRARIES})
target_include_directories(bert_weight PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(bert_weight PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

add_library(quant_bert_weight STATIC quant_bert_weight.cc ${QUANT_BERT_PROTO_SRC}
                               ${QUANT_BERT_PROTO_HEADER})
target_link_libraries(quant_bert_weight PUBLIC utils ${Protobuf_LIBRARIES})
target_include_directories(quant_bert_weight PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(quant_bert_weight PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

add_library(bert_crf_weight STATIC bert_crf_weight.cc ${BERT_CRF_PROTO_SRC}
                                   ${BERT_CRF_PROTO_HEADER})
target_link_libraries(bert_crf_weight PUBLIC utils ${Protobuf_LIBRARIES})
target_include_directories(bert_crf_weight PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(bert_crf_weight PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
