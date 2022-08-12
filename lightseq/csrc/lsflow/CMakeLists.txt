cmake_minimum_required(VERSION 3.10)
project(LightseqProtoType LANGUAGES CXX)
find_package(Threads REQUIRED)
set(CMAKE_CXX_STANDARD 14)

include_directories(includes)

add_definitions(-DDEBUG)

add_executable(lsflow_bin example/main.cpp)

add_library(node SHARED node.cpp)

add_library(context SHARED context.cpp)

add_library(manager SHARED manager.cpp)

add_library(layer SHARED layer.cpp)

add_library(tensor SHARED tensor.cpp)

target_link_libraries(lsflow_bin PUBLIC node)
target_link_libraries(lsflow_bin PUBLIC context)
target_link_libraries(lsflow_bin PUBLIC manager)
target_link_libraries(lsflow_bin PUBLIC layer)
target_link_libraries(lsflow_bin PUBLIC tensor)
