########lightseq_hip构建######

#1、设置HIP env,参考如下：
export C_INCLUDE_PATH=${ROCM_PATH}/rocblas/include:${ROCM_PATH}/rocrand/include/:${ROCM_PATH}/hiprand/include:${ROCM_PATH}/hip/include/hip:$ROCM_PATH/hip/include/hip/hsa_detail:$ROCM_PATH/hipcub/include/hipcub:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$C_INCLUDE_PATH
export LD_LIBRART_PATH=${ROCM_PATH}/rocblas/lib/:${ROCM_PATH}/rocrand/lib:$LD_LIBRART_PATH
export LIBRARY_PATH=${ROCM_PATH}/rocblas/lib/:$LIBRARY_PATH


#2、编译ls
source `pwd`/shell/env_build.sh
verbose=1 ENABLE_FP32=1 ENABLE_DEBUG=1 CXX=hipcc CC=hipcc python3 setup.py install bdist_wheel

#编译后在dist下生成whl文件

#3、简单用例验证
同cuda环境运行方式，如需在指定卡上运行，可使用HIP_VISIBLE_DEVICES指定。使用ls训练transformer示例：

export HIP_VISIBLE_DEVICES=0
lightseq-train /public/DL_DATA/wmt14_en_de_joined_dict  \
    --task translation \
    --arch ls_transformer_wmt_en_de_big_t2t --share-decoder-input-output-embed \
    --optimizer ls_adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --weight-decay 0.0001 \
    --criterion ls_label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --fp16
