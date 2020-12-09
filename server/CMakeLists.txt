cmake_minimum_required(VERSION 3.18)

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

protobuf_generate_cpp(CONFIG_PROTO_SRC CONFIG_PROTO_HEADER model_config.proto)
configure_file(libserver.ldscript libserver.ldscript COPYONLY)

add_library(
  transformer_server SHARED transformer_server.cc.cu custom.h
                            ${CONFIG_PROTO_SRC} ${CONFIG_PROTO_HEADER})
target_link_libraries(transformer_server PUBLIC transformer_model)
target_include_directories(transformer_server
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(transformer_server
                           INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(
  transformer_server PROPERTIES LINK_DEPENDS
                                ${CMAKE_CURRENT_BINARY_DIR}/libserver.ldscript)
set_target_properties(
  transformer_server PROPERTIES LINK_FLAGS
                                "-Wl,--version-script libserver.ldscript")

add_library(
  transformer_generate_server SHARED generate_server.cc.cu custom.h
                                     ${CONFIG_PROTO_SRC} ${CONFIG_PROTO_HEADER})
target_link_libraries(transformer_generate_server PUBLIC transformer_model)
target_include_directories(transformer_generate_server
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(transformer_generate_server
                           INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(
  transformer_generate_server
  PROPERTIES LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libserver.ldscript)
set_target_properties(
  transformer_generate_server
  PROPERTIES LINK_FLAGS "-Wl,--version-script libserver.ldscript")

add_library(gpt_lm_server SHARED gptlm_server.cc.cu custom.h
                                 ${CONFIG_PROTO_SRC} ${CONFIG_PROTO_HEADER})
target_link_libraries(gpt_lm_server PUBLIC gpt_model)
target_include_directories(gpt_lm_server INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(gpt_lm_server INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(
  gpt_lm_server PROPERTIES LINK_DEPENDS
                           ${CMAKE_CURRENT_BINARY_DIR}/libserver.ldscript)
set_target_properties(
  gpt_lm_server PROPERTIES LINK_FLAGS "-Wl,--version-script libserver.ldscript")

add_library(
  gpt_generate_server SHARED gpt_generate_server.cc.cu custom.h
                             ${CONFIG_PROTO_SRC} ${CONFIG_PROTO_HEADER})
target_link_libraries(gpt_generate_server PUBLIC gpt_model)
target_include_directories(gpt_generate_server
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(gpt_generate_server
                           INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(
  gpt_generate_server PROPERTIES LINK_DEPENDS
                                 ${CMAKE_CURRENT_BINARY_DIR}/libserver.ldscript)
set_target_properties(
  gpt_generate_server PROPERTIES LINK_FLAGS
                                 "-Wl,--version-script libserver.ldscript")

add_library(
  decoder_generate_server SHARED decoder_generate_server.cc.cu custom.h
                                 ${CONFIG_PROTO_SRC} ${CONFIG_PROTO_HEADER})
target_link_libraries(decoder_generate_server PUBLIC transformer_model)
target_include_directories(decoder_generate_server
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(decoder_generate_server
                           INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(
  decoder_generate_server
  PROPERTIES LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libserver.ldscript)
set_target_properties(
  decoder_generate_server PROPERTIES LINK_FLAGS
                                     "-Wl,--version-script libserver.ldscript")
