cmake_minimum_required(VERSION 3.18)

add_executable(bert_example bert_example.cc)
target_link_libraries(bert_example PUBLIC liblightseq)
