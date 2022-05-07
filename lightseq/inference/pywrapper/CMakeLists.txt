cmake_minimum_required(VERSION 3.18)

pybind11_add_module(
  lightseq
  MODULE
  wrapper.cc
  transformer.cc
  gpt.cc
  bert.cc
  quant_transformer.cc
  quant_bert.cc
  quant_gpt.cc
  moe.cc
  vit.cc)
target_link_libraries(lightseq PUBLIC gpt_model)
target_link_libraries(lightseq PUBLIC bert_model)
target_link_libraries(lightseq PUBLIC transformer_model)
target_link_libraries(lightseq PUBLIC quant_transformer_model)
target_link_libraries(lightseq PUBLIC quant_bert_model)
target_link_libraries(lightseq PUBLIC quant_gpt_model)
target_link_libraries(lightseq PUBLIC moe_model)
target_link_libraries(lightseq PUBLIC vit_model)
set_target_properties(lightseq PROPERTIES OUTPUT_NAME inference)

add_library(
  liblightseq SHARED
  transformer.cc
  gpt.cc
  bert.cc
  quant_transformer.cc
  quant_bert.cc
  quant_gpt.cc
  moe.cc
  vit.cc)
target_link_libraries(liblightseq PUBLIC transformer_model)
target_link_libraries(liblightseq PUBLIC quant_transformer_model)
target_link_libraries(liblightseq PUBLIC quant_bert_model)
target_link_libraries(liblightseq PUBLIC quant_gpt_model)
target_link_libraries(liblightseq PUBLIC gpt_model)
target_link_libraries(liblightseq PUBLIC bert_model)
target_link_libraries(liblightseq PUBLIC moe_model)
target_link_libraries(liblightseq PUBLIC vit_model)
target_link_options(liblightseq PUBLIC $<DEVICE_LINK:-Xcompiler
                    -fvisibility=hidden>)
target_include_directories(liblightseq PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
