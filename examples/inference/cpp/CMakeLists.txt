cmake_minimum_required(VERSION 3.18)

add_executable(transformer_example transformer_example.cc)
target_link_libraries(transformer_example PUBLIC liblightseq)

add_executable(quant_transformer_example quant_transformer_example.cc)
target_link_libraries(quant_transformer_example PUBLIC liblightseq)

add_executable(bert_example bert_example.cc)
target_link_libraries(bert_example PUBLIC liblightseq)

add_executable(quant_bert_example quant_bert_example.cc)
target_link_libraries(quant_bert_example PUBLIC liblightseq)

add_executable(gpt_example gpt_example.cc)
target_link_libraries(gpt_example PUBLIC liblightseq)

add_executable(quant_gpt_example quant_gpt_example.cc)
target_link_libraries(quant_gpt_example PUBLIC liblightseq)

add_executable(transformer_decoder_example decoder_example.cc.cu)
target_link_libraries(transformer_decoder_example PUBLIC transformer_model)

add_executable(vit_example vit_example.cc)
target_link_libraries(vit_example PUBLIC liblightseq)

add_executable(quant_vit_example quant_vit_example.cc)
target_link_libraries(quant_vit_example PUBLIC liblightseq)
