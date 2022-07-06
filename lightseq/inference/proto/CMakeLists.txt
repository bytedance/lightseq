cmake_minimum_required(VERSION 3.18)

# (default) use C API for HDF5 library
find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

protobuf_generate_cpp(GPT_PROTO_SRC GPT_PROTO_HEADER gpt.proto)
protobuf_generate_cpp(Q_GPT_PROTO_SRC Q_GPT_PROTO_HEADER quant_gpt.proto)
protobuf_generate_cpp(BERT_PROTO_SRC BERT_PROTO_HEADER bert.proto)
protobuf_generate_cpp(Q_BERT_PROTO_SRC Q_BERT_PROTO_HEADER quant_bert.proto)
protobuf_generate_cpp(Q_TRANSFORMER_PROTO_SRC Q_TRANSFORMER_PROTO_HEADER
                      quant_transformer.proto)
protobuf_generate_cpp(TRANSFORMER_PROTO_SRC TRANSFORMER_PROTO_HEADER
                      transformer.proto)
protobuf_generate_cpp(MOE_PROTO_SRC MOE_PROTO_HEADER moe.proto)
protobuf_generate_cpp(VIT_PROTO_SRC VIT_PROTO_HEADER vit.proto)
protobuf_generate_cpp(T5_PROTO_SRC T5_PROTO_HEADER t5.proto)

add_library(gpt_weight STATIC gpt_weight.cc ${GPT_PROTO_SRC}
                              ${GPT_PROTO_HEADER})
target_link_libraries(gpt_weight PUBLIC utils ${Protobuf_LIBRARIES}
                                        ${HDF5_LIBRARIES})
target_include_directories(gpt_weight PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(gpt_weight PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

add_library(quant_gpt_weight STATIC quant_gpt_weight.cc ${Q_GPT_PROTO_SRC}
                                    ${Q_GPT_PROTO_HEADER})
target_link_libraries(quant_gpt_weight PUBLIC utils ${Protobuf_LIBRARIES}
                                              ${HDF5_LIBRARIES})
target_include_directories(quant_gpt_weight PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(quant_gpt_weight PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

add_library(bert_weight STATIC bert_weight.cc ${BERT_PROTO_SRC}
                               ${BERT_PROTO_HEADER})
target_link_libraries(bert_weight PUBLIC utils ${Protobuf_LIBRARIES})
target_include_directories(bert_weight PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(bert_weight PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

add_library(quant_bert_weight STATIC quant_bert_weight.cc ${Q_BERT_PROTO_SRC}
                                     ${Q_BERT_PROTO_HEADER})
target_link_libraries(quant_bert_weight PUBLIC utils ${Protobuf_LIBRARIES})
target_include_directories(quant_bert_weight PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(quant_bert_weight PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

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

add_library(moe_weight STATIC moe_weight.cc ${MOE_PROTO_SRC}
                              ${MOE_PROTO_HEADER})
target_link_libraries(moe_weight PUBLIC utils ${Protobuf_LIBRARIES}
                                        ${HDF5_LIBRARIES})
target_include_directories(moe_weight PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(moe_weight PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

add_library(vit_weight STATIC vit_weight.cc ${VIT_PROTO_SRC}
                              ${VIT_PROTO_HEADER})
target_link_libraries(vit_weight PUBLIC utils ${Protobuf_LIBRARIES})
target_include_directories(vit_weight PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(vit_weight PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

add_library(t5_weight STATIC t5_weight.cc ${T5_PROTO_SRC} ${T5_PROTO_HEADER})
target_link_libraries(t5_weight PUBLIC utils ${Protobuf_LIBRARIES}
                                       ${HDF5_LIBRARIES})
target_include_directories(t5_weight PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(t5_weight PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
