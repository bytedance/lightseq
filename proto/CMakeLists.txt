cmake_minimum_required(VERSION 3.18)

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
# set(weight_files gpt_weight.cc transformer_weight.cc)

protobuf_generate_cpp(GPT_PROTO_SRC GPT_PROTO_HEADER gpt.proto)
# add_library(gpt_proto STATIC ${GPT_PROTO_HEADER} ${GPT_PROTO_SRC})
protobuf_generate_cpp(TRANSFORMER_PROTO_SRC TRANSFORMER_PROTO_HEADER
                      transformer.proto)

add_library(gpt_weight STATIC gpt_weight.cc ${GPT_PROTO_SRC}
                              ${GPT_PROTO_HEADER})
target_link_libraries(gpt_weight PUBLIC utils ${Protobuf_LIBRARIES})
target_include_directories(gpt_weight INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(gpt_weight INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

add_library(
  transformer_weight STATIC transformer_weight.cc ${TRANSFORMER_PROTO_SRC}
                            ${TRANSFORMER_PROTO_HEADER})
target_link_libraries(transformer_weight PUBLIC utils ${Protobuf_LIBRARIES})
target_include_directories(transformer_weight
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(transformer_weight
                           INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
