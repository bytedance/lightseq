add_library(liblightseq SHARED bert.cc bert_crf.cc transformer.cc)

target_link_libraries(liblightseq PUBLIC lightseq_layers utils)

target_link_libraries(liblightseq PUBLIC bert_weight bert_crf_weight
                                         transformer_weight)

target_link_options(liblightseq PUBLIC $<DEVICE_LINK:-Xcompiler
                    -fvisibility=hidden>)

target_include_directories(liblightseq PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(liblightseq PROPERTIES OUTPUT_NAME lightseq)
