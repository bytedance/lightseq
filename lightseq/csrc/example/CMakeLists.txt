cmake_minimum_required(VERSION 3.18)

add_executable(bert_example bert_example.cc)
target_link_libraries(bert_example PUBLIC liblightseq)

add_executable(big_gpt_example big_gpt_example.cc)
target_link_libraries(big_gpt_example PUBLIC liblightseq)

add_executable(transformer_example transformer_example.cc)
target_link_libraries(transformer_example PUBLIC liblightseq)

add_executable(gpt_example gpt_example.cc)
target_link_libraries(gpt_example PUBLIC liblightseq)

add_executable(llama_example llama_example.cc)
target_link_libraries(llama_example PUBLIC liblightseq)
