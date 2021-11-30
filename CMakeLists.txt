cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(LightSeq LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 61 70 75)
find_package(CUDA 10.1 REQUIRED)

option(FP16_MODE "inference with fp16" OFF)
option(DEBUG_MODE "debug computation result" OFF)

set(Protobuf_USE_STATIC_LIBS ON)

set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)

# setting compiler flags
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -Xcompiler -Wall")
set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
set(CMAKE_CUDA_FLAGS_RELEASE
    "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O2")

set(COMMON_HEADER_DIRS ${PROJECT_SOURCE_DIR} ${CUDA_PATH}/include)
set(COMMON_LIB_DIRS ${CUDA_PATH}/lib64)

include_directories(${COMMON_HEADER_DIRS})

link_directories(${COMMON_LIB_DIRS})

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

add_subdirectory(lightseq/inference/kernels)
add_subdirectory(lightseq/inference/tools)
add_subdirectory(lightseq/inference/proto)
add_subdirectory(lightseq/inference/model)
add_subdirectory(examples/inference/cpp)
add_subdirectory(3rdparty/pybind11)
add_subdirectory(lightseq/inference/pywrapper)
add_subdirectory(lightseq/inference/server)
