cmake_minimum_required(VERSION 3.8)

find_package(Protobuf REQUIRED)

set(weight_files
  gpt_weight.cc
  transformer_weight.cc
)

protobuf_generate_cpp(GPT_PROTO_SRC GPT_PROTO_HEADER gpt.proto)
add_library(gpt_proto STATIC ${GPT_PROTO_HEADER} ${GPT_PROTO_SRC})
protobuf_generate_cpp(TRANSFORMER_PROTO_SRC TRANSFORMER_PROTO_HEADER transformer.proto)
add_library(transformer_proto STATIC ${TRANSFORMER_PROTO_HEADER} ${TRANSFORMER_PROTO_SRC})
target_include_directories(gpt_proto
                           INTERFACE ${CMAKE_CURRENT_BINARY_DIR}
                           )
target_include_directories(transformer_proto
                           INTERFACE ${CMAKE_CURRENT_BINARY_DIR}
                           )                           

add_library(gpt_weight STATIC gpt_weight.cc)
target_link_libraries(gpt_weight PUBLIC utils gpt_proto )
target_include_directories(gpt_weight
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
                           ${CMAKE_CURRENT_BINARY_DIR}
                           )

add_library(transformer_weight STATIC transformer_weight.cc)
target_link_libraries(transformer_weight PUBLIC utils transformer_proto)
target_include_directories(transformer_weight
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
                           ${CMAKE_CURRENT_BINARY_DIR}
                           )