add_executable(lightseq_prototype_bin test_example.cpp)

include_directories(../core/includes)

target_link_libraries(lightseq_prototype_bin PUBLIC core)
