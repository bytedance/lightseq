cmake_minimum_required(VERSION 3.18)

# load HDF5 package
set(HDF5_USE_STATIC_LIBRARIES ON)
# (default) use C API for HDF5 library
find_package(HDF5 REQUIRED)

add_library(utils STATIC util.cc.cu)
target_include_directories(utils PUBLIC ${HDF5_INCLUDE_DIRS})
target_include_directories(utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(utils PRIVATE ${HDF5_LIBRARIES})
