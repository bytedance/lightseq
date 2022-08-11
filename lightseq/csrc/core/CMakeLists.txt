find_package(Threads REQUIRED)

include_directories(includes)

add_library(node SHARED node.cpp)

add_library(context SHARED context.cpp)

add_library(manager SHARED manager.cpp)

add_library(layer SHARED layer.cpp)

add_library(tensor SHARED tensor.cpp)
