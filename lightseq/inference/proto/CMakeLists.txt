cmake_minimum_required(VERSION 3.18)

# (default) use C API for HDF5 library
find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

protobuf_generate_cpp(GPT_PROTO_SRC GPT_PROTO_HEADER gpt.proto)
protobuf_generate_cpp(BERT_PROTO_SRC BERT_PROTO_HEADER bert.proto)
protobuf_generate_cpp(Q_TRANSFORMER_PROTO_SRC Q_TRANSFORMER_PROTO_HEADER
                      transformer_quant.proto)
protobuf_generate_cpp(TRANSFORMER_PROTO_SRC TRANSFORMER_PROTO_HEADER
                      transformer.proto)

add_library(gpt_weight STATIC gpt_weight.cc ${GPT_PROTO_SRC}
                              ${GPT_PROTO_HEADER})
target_link_libraries(gpt_weight PUBLIC utils ${Protobuf_LIBRARIES}
                                        ${HDF5_LIBRARIES})
target_include_directories(gpt_weight PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(gpt_weight PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

add_library(bert_weight STATIC bert_weight.cc ${BERT_PROTO_SRC}
                               ${BERT_PROTO_HEADER})
target_link_libraries(bert_weight PUBLIC utils ${Protobuf_LIBRARIES})
target_include_directories(bert_weight PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(bert_weight PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

add_library(
  transformer_weight STATIC transformer_weight.cc ${TRANSFORMER_PROTO_SRC}
                            ${TRANSFORMER_PROTO_HEADER})
target_link_libraries(transformer_weight PUBLIC utils ${Protobuf_LIBRARIES}
                                                ${HDF5_LIBRARIES})
target_include_directories(transformer_weight
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(transformer_weight
                           PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

add_library(
  quant_transformer_weight STATIC
  quant_transformer_weight.cc ${Q_TRANSFORMER_PROTO_SRC}
  ${Q_TRANSFORMER_PROTO_HEADER})
target_link_libraries(quant_transformer_weight
                      PUBLIC utils ${Protobuf_LIBRARIES} ${HDF5_LIBRARIES})
target_include_directories(quant_transformer_weight
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(quant_transformer_weight
                           PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
