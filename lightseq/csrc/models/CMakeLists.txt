add_library(
  liblightseq SHARED
  bert.cc
  bert_crf.cc
  transformer.cu
  gpt.cc
  llama.cc
  model_util.cc
  model_base.cc)

target_link_libraries(liblightseq PUBLIC lightseq_layers)

target_link_libraries(liblightseq PUBLIC weight_lib)

target_link_options(liblightseq PUBLIC $<DEVICE_LINK:-Xcompiler
                    -fvisibility=hidden>)

target_include_directories(liblightseq PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(liblightseq PROPERTIES OUTPUT_NAME lightseq)

# add_executable(test_example test_layer.cc) target_link_libraries(test_example
# PUBLIC liblightseq)
