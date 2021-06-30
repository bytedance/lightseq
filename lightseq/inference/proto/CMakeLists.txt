cmake_minimum_required(VERSION 3.18)

# load HDF5 package
set(HDF5_USE_STATIC_LIBRARIES ON)
# (default) use C API for HDF5 library
find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

protobuf_generate_cpp(GPT_PROTO_SRC GPT_PROTO_HEADER gpt.proto)
protobuf_generate_cpp(TRANSFORMER_PROTO_SRC TRANSFORMER_PROTO_HEADER
                      transformer.proto)

add_library(gpt_weight STATIC gpt_weight.cc ${GPT_PROTO_SRC} ${GPT_PROTO_HEADER})
target_link_libraries(gpt_weight PUBLIC utils ${Protobuf_LIBRARIES} ${HDF5_LIBRARIES})
target_include_directories(gpt_weight INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(gpt_weight INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

add_library(transformer_weight STATIC transformer_weight.cc ${TRANSFORMER_PROTO_SRC} ${TRANSFORMER_PROTO_HEADER})
target_link_libraries(transformer_weight PUBLIC utils ${Protobuf_LIBRARIES} ${HDF5_LIBRARIES})
target_include_directories(transformer_weight INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(transformer_weight INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
