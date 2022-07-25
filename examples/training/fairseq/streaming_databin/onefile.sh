#!/usr/bin/env bash
set -ex
THIS_DIR=$(dirname $(readlink -f $0))
cd $THIS_DIR/../../../..

databin=examples/training/fairseq/databin
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/duanrenchong/datasets/en-fr/onefile_databin $databin

#hdfs://haruna/home/byte_arnold_lq_mlnlc/user/duanrenchong/datasets/en-fr/databin
# lightseq-train /opt/tiger/begin/lightseq/examples/training/fairseq/databin: \

lightseq-train $databin \
    --task translation \
    --source-lang en --target-lang fr \
    --arch ls_transformer --share-decoder-input-output-embed \
    --optimizer ls_adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --weight-decay 0.0001 \
    --criterion ls_label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8192 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --fp16
