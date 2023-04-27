set(operator_files
    beam_search_topk.cu
    bias_act_dropout.cpp
    bias_add_transform_20314.cpp
    bias_dropout_residual.cpp
    concat3_dim1.cpp
    crf.cpp
    dropout.cpp
    launch_dec_emb_op.cpp
    launch_enc_emb.cpp
    launch_gpt_emb.cpp
    layer_normalize.cpp
    split_head_op.cpp
    linear.cpp
    rotary_position_qk.cpp
    sampling.cc.cu
    softmax.cpp
    strided_batch_gemm.cpp
    transform_0213.cpp)

add_library(lightseq_operators STATIC ${operator_files})
target_link_libraries(lightseq_operators PUBLIC lsflow)
target_include_directories(lightseq_operators PUBLIC includes)
