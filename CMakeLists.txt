cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(LightSeq LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES
    60
    61
    70
    75
    80
    86
    87)
find_package(CUDA 11.6 REQUIRED)

option(FP16_MODE "inference with fp16" OFF)
option(DEBUG_MODE "debug computation result" OFF)
option(DYNAMIC_API "build dynamic lightseq api library" OFF)
option(USE_TRITONBACKEND "build tritonbackend for lightseq" OFF)

set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)

# setting compiler flags
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -Xcompiler -Wall")

if(DYNAMIC_API)
  # dynamic link to cuda libraries and protobuf
  set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared")
  set(HDF5_USE_STATIC_LIBRARIES OFF)
else()
  # static link to cuda libraries and protobuf
  set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")
  set(HDF5_USE_STATIC_LIBRARIES ON)
endif()

set(Protobuf_USE_STATIC_LIBS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(COMMON_HEADER_DIRS ${PROJECT_SOURCE_DIR} ${CUDA_PATH}/include)
set(COMMON_LIB_DIRS ${CUDA_PATH}/lib64)

include_directories(${COMMON_HEADER_DIRS})
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/3rdparty/cub)

link_directories(${COMMON_LIB_DIRS})

add_compile_options(-Wno-unknown-pragmas)

if(FP16_MODE)
  add_definitions(-DFP16_MODE)
  message(STATUS "Build using fp16 precision")
else()
  message(STATUS "Build using fp32 precision")
endif()

if(DEBUG_MODE)
  add_definitions(-DDEBUG_RESULT)
  message(STATUS "Debug computation result")
endif()

add_subdirectory(3rdparty/pybind11)
add_subdirectory(lightseq/inference/kernels)
add_subdirectory(lightseq/inference/tools)
add_subdirectory(lightseq/inference/proto)
add_subdirectory(lightseq/inference/model)
add_subdirectory(lightseq/inference/pywrapper)
add_subdirectory(lightseq/inference/server)
if(USE_TRITONBACKEND)
  add_subdirectory(lightseq/inference/triton_backend)
endif()

# add_subdirectory(examples/inference/cpp)
