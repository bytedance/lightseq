cmake_minimum_required(VERSION 3.10)
project(LightseqProtoType LANGUAGES CXX)
find_package(Threads REQUIRED)

add_subdirectory(core)

add_subdirectory(example)
