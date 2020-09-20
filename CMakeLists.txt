cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(FasterTransformer LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 61 70 75)
find_package(CUDA 11 REQUIRED)

option(FP16_MODE "inference with fp16" OFF)

set(Protobuf_USE_STATIC_LIBS ON)

set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)

# setting compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")
set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++14")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -fPIC")
set(CMAKE_CUDA_FLAGS_RELEASE
    "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O2 -Xcompiler -fPIC")

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

add_subdirectory(kernels)
add_subdirectory(tools)
add_subdirectory(proto)
add_subdirectory(model)
add_subdirectory(example)
add_subdirectory(3rdparty/pybind11)
add_subdirectory(pywrapper)
