cmake_minimum_required(VERSION 3.18)

add_executable(gpt_generation_example gpt_generation.cc.cu)
# set_target_properties(gpt_generation_example PROPERTIES
# CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(gpt_generation_example PUBLIC gpt_model)

add_executable(gptlm_example gptlm_example.cc.cu)
target_link_libraries(gptlm_example PUBLIC gpt_model)
# set_target_properties(gptlm_example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add_executable(transformer_example transformer_example.cc.cu)
# target_link_libraries(transformer_example PUBLIC transformer_model)
# set_target_properties(transformer_example PROPERTIES
# CUDA_SEPARABLE_COMPILATION ON)

add_executable(transformer_example transformer_example.cc.cu)
target_link_libraries(transformer_example PUBLIC lightseq)
# target_compile_options(transformer_example BEFORE PUBLIC
# -forward-unknown-to-host-compiler)
# target_include_directories(transformer_example PRIVATE ${PYBIND11_INCLUDE_DIR}
# PRIVATE ${pybind11_INCLUDE_DIR} PRIVATE ${PYTHON_INCLUDE_DIRS})
# set_target_properties(transformer_example PROPERTIES
# CUDA_SEPARABLE_COMPILATION ON)

add_executable(transformer_generate_example transformer_generate_example.cc.cu)
target_link_libraries(transformer_generate_example PUBLIC transformer_model)
# set_target_properties(transformer_generate_example PROPERTIES
# CUDA_SEPARABLE_COMPILATION ON)

add_executable(transformer_decoder_example decoder_example.cc.cu)
target_link_libraries(transformer_decoder_example PUBLIC transformer_model)
# set_target_properties(transformer_decoder_example PROPERTIES
# CUDA_SEPARABLE_COMPILATION ON)
