cmake_minimum_required(VERSION 3.18)

add_library(utils STATIC util.cc.cu)
target_include_directories(utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
