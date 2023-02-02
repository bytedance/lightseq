cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(LightSeq LANGUAGES C CXX CUDA)

option(USE_NEW_ARCH "inference with new arch" OFF)
option(FP16_MODE "inference with fp16" OFF)
option(DEBUG_MODE "debug computation result" OFF)
option(MEM_DEBUG "debug memory message" OFF)
option(DYNAMIC_API "build dynamic lightseq api library" OFF)
option(USE_TRITONBACKEND "build tritonbackend for lightseq" OFF)
option(USE_PYBIND "build lightseq with pybind interface" OFF)

if(USE_NEW_ARCH)
  add_definitions(-DNEW_ARCH)

  set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 87)
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

  set(DEVICE_ARCHITECTURES_LIST cuda x86 arm)
  list(FIND DEVICE_ARCHITECTURES_LIST ${DEVICE_ARCH} DEVICE_INDEX)

  if(DEVICE_INDEX EQUAL 0)
    add_definitions(-DLIGHTSEQ_cuda)
    set(DEVICE_ARCHITECTURE cuda)
  elseif(DEVICE_INDEX EQUAL 1)
    add_definitions(-DLIGHTSEQ_x86)
    set(DEVICE_ARCHITECTURE x86)
  elseif(DEVICE_INDEX EQUAL 2)
    add_definitions(-DLIGHTSEQ_arm)
    set(DEVICE_ARCHITECTURE arm)
  else()
    message(
      WARNING "compiled with -DDEVICE_ARCHITECTURE=${DEVICE_ARCHITECTURE}")
    message(
      FATAL_ERROR
        "-DDEVICE_ARCHITECTURE=\$\{device\} must in value of list [${DEVICE_ARCHITECTURES_LIST}]"
    )
    return()
  endif()
  message(STATUS "compile with device ${DEVICE_ARCHITECTURE} ${index}")

  if(DEVICE_INDEX GREATER 0 AND FP16_MODE)
    message(FATAL_ERROR "CPU device does not have fp16 version")
    return()
  endif()

  if(DEBUG_MODE)
    add_definitions(-DDEBUG_MODE)
    set(MEM_DEBUG ON)
    message(STATUS "Build using debug mode")
  endif()

  if(MEM_DEBUG)
    add_definitions(-DMEM_DEBUG)
    message(STATUS "Build using memory debug")
  endif()

  if(FP16_MODE)
    add_definitions(-DFP16_MODE)
    message(STATUS "Build using fp16 precision")
  else()
    message(STATUS "Build using fp32 precision")
  endif()

  set(COMMON_HEADER_DIRS
      ${PROJECT_SOURCE_DIR}
      ${CUDA_PATH}/include
      lightseq/csrc/kernels/${DEVICE_ARCHITECTURE}/includes
      lightseq/csrc/layers_new/includes
      lightseq/csrc/lsflow/includes
      lightseq/csrc/models/includes
      lightseq/csrc/ops_new/includes
      lightseq/csrc/proto/includes)

  set(COMMON_LIB_DIRS ${CUDA_PATH}/lib64)

  include_directories(${COMMON_HEADER_DIRS})
  include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/3rdparty/cub)

  link_directories(${COMMON_LIB_DIRS})

  add_subdirectory(3rdparty/pybind11)
  # add_subdirectory(lightseq/csrc/tools)
  add_subdirectory(lightseq/csrc/kernels/${DEVICE_ARCHITECTURE})
  add_subdirectory(lightseq/csrc/layers_new)
  add_subdirectory(lightseq/csrc/lsflow)
  add_subdirectory(lightseq/csrc/models)
  add_subdirectory(lightseq/csrc/ops_new)
  add_subdirectory(lightseq/csrc/proto)
  # add_subdirectory(lightseq/csrc/example)
  if(USE_PYBIND)
    add_subdirectory(lightseq/csrc/pybind)
  endif()
  if(USE_TRITONBACKEND)
    add_subdirectory(lightseq/csrc/triton_backend)
  endif()

else()

  find_package(CUDA 11 REQUIRED)

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

  set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86)

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

endif()
