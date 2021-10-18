cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(LightSeq LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75)
find_package(CUDA 10.1 REQUIRED)

option(FP16_MODE "inference with fp16" OFF)
option(DEBUG_MODE "debug computation result" OFF)
option(BUILD_PYTHON "build for python api" OFF)
option(DYNAMIC_API "build dynamic lightseq api library" OFF)

if(FP16_MODE)
  add_compile_definitions(FP16_MODE)
  message(STATUS "Build using fp16 precision")
else()
  message(STATUS "Build using fp32 precision")
endif()

if(DEBUG_MODE)
  add_compile_definitions(DEBUG_RESULT)
  message(STATUS "Debug computation result")
endif()

if(BUILD_PYTHON)
  add_compile_definitions(ENABLE_PYTHON)
  set(DYNAMIC_API OFF)
  message(STATUS "Build for python wrapper")
endif()

set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)

# setting compiler flags
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -Xcompiler -Wall")

if(DYNAMIC_API)
  # dynamic link to cuda libraries and protobuf
  set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared")
  set(Protobuf_USE_STATIC_LIBS OFF)
else()
  # static link to cuda libraries and protobuf
  set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")
  set(Protobuf_USE_STATIC_LIBS ON)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O2")

set(COMMON_HEADER_DIRS ${PROJECT_SOURCE_DIR} ${CUDA_PATH}/include)
set(COMMON_LIB_DIRS ${CUDA_PATH}/lib64)

include_directories(${COMMON_HEADER_DIRS})
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/3rdparty/cub)

link_directories(${COMMON_LIB_DIRS})

add_subdirectory(3rdparty/pybind11)
add_subdirectory(lightseq/inference/kernels)
add_subdirectory(lightseq/inference/tools)
add_subdirectory(lightseq/inference/proto)
add_subdirectory(lightseq/inference/model)
add_subdirectory(lightseq/inference/pywrapper)
add_subdirectory(lightseq/inference/server)

if(NOT BUILD_PYTHON)
  add_subdirectory(examples/inference/cpp)
endif()
