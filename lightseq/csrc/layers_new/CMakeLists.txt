set(layers_files
    feed_forward_layer.cpp
    linear_layer.cpp
    # llama_attention_layer.cpp
    # llama_mlp_layer.cpp
    generator_layer.cpp
    gpt_attention_layer.cpp
    gpt_layer.cpp
    multihead_attention_layer.cpp
    transformer_encoder_layer.cpp
    dec_enc_attention_layer.cpp
    dec_self_attention_layer.cpp
    transformer_decoder_layer.cpp
    crf_layer.cpp
    encdec_kv_layer.cpp
    sample_layer.cpp
    sdpa_layer.cpp)

add_library(lightseq_layers STATIC ${layers_files})
target_link_libraries(lightseq_layers PUBLIC lightseq_operators lsflow)
target_include_directories(lightseq_layers PUBLIC includes)
