
set(operator_files 
    bias_act_dropout.cpp
    bias_add_transform_20314.cpp
    bias_dropout_residual.cpp
    dropout.cpp
    feed_forward.cpp
    launch_enc_emb.cpp
    normalize_layer.cpp
    softmax.cpp
    strided_batch_gemm.cpp
    transform_0213.cpp
)

add_library(lightseq_operators STATIC ${operator_files})
target_link_libraries(lightseq_operators PUBLIC cuda_kernels -lcublas)
target_include_directories(lightseq_operators PUBLIC includes)
