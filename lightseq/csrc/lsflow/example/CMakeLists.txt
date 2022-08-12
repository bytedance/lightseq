add_executable(lsflow_bin test_example.cpp)

include_directories(../includes)

target_link_libraries(lsflow_bin PUBLIC node)
target_link_libraries(lsflow_bin PUBLIC context)
target_link_libraries(lsflow_bin PUBLIC manager)
target_link_libraries(lsflow_bin PUBLIC layer)
target_link_libraries(lsflow_bin PUBLIC tensor)
