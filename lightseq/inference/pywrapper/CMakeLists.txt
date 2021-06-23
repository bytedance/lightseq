cmake_minimum_required(VERSION 3.18)

pybind11_add_module(lightseq MODULE wrapper.cc)
target_link_libraries(lightseq PUBLIC transformer_model)
target_link_libraries(lightseq PUBLIC gpt_model)
set_target_properties(lightseq PROPERTIES OUTPUT_NAME lightseq/inference)
