cmake_minimum_required(VERSION 3.8)

add_library(utils STATIC util.cc.cu)
target_include_directories(utils
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
                           )
target_link_libraries(utils PUBLIC -lcublas -lcudart)
set_target_properties(utils PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)