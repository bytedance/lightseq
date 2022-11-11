set(operator_files
    bias_act_dropout.cpp
    bias_add_transform_20314.cpp
    bias_dropout_residual.cpp
    crf.cpp
    dropout.cpp
    concat3_dim1.cpp
    launch_enc_emb.cpp
    launch_dec_emb_op.cpp
    layer_normalize.cpp
    linear.cpp
    softmax.cpp
    strided_batch_gemm.cpp
    transform_0213.cpp
    beam_search_topk.cu)

add_library(lightseq_operators STATIC ${operator_files})
target_link_libraries(lightseq_operators PUBLIC cuda_kernels -lcublas lsflow)
target_include_directories(lightseq_operators PUBLIC includes)
