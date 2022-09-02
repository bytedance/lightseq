add_library(liblightseq SHARED bert.cc)

target_link_libraries(liblightseq PUBLIC lightseq_layers utils)

target_link_libraries(liblightseq PUBLIC bert_weight)

target_link_options(liblightseq PUBLIC $<DEVICE_LINK:-Xcompiler
                    -fvisibility=hidden>)

target_include_directories(liblightseq PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
