#!/usr/bin/env bash
set -ex
THIS_DIR=$(dirname $(readlink -f $0))
cd $THIS_DIR/../../..

if [ ! -d "/tmp/wmt14" ]; then
    echo "Downloading dataset"
    hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/duanrenchong/datasets/en-fr/onefile_databin /tmp/wmt14
fi

lightseq-train /tmp/wmt14/ \
    --task translation \
    --save-dir int4 \
    --share-decoder-input-output-embed \
    --optimizer ls_adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --warmup-updates 4000 \
    --criterion ls_label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8192 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --fp16 \
    --use-torch-layer \
    --enable-quant \
    --finetune-from-model fp16/checkpoint_best.pt \
    --quant-mode qat --quant-bits 4 \
    --n-gpus-intwhat 16 \
    --max-epoch 160 --keep-last-epochs 1 --smooth-avg-update 200 $@

# --weight-decay 0.0001

# --arch ls_transformer --lr 5e-4 --lr-scheduler inverse_sqrt 
# --arch ls_transformer --lr 5e-4 --lr-scheduler polynomial_decay --total-num-update 150000 --end-learning-rate 1e-6

# hdfs dfs -put quant_scape_2/* hdfs://haruna/home/byte_arnold_lq_mlnlc/user/duanrenchong/pretrain_model/wmt14en-fr/int8_torch/