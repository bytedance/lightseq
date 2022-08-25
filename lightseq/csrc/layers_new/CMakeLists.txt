
set(layers_files
    feed_forward_layer.cpp
    multihead_attention_layer.cpp
    transformer_encoder_layer.cpp
)

add_library(lightseq_layers STATIC ${layers_files})
target_link_libraries(lightseq_layers PUBLIC lightseq_operators)
target_include_directories(lightseq_layers PUBLIC includes)
