add_executable(lsflow_bin main.cpp)

include_directories(../includes)

target_link_libraries(lsflow_bin PUBLIC lsflow)