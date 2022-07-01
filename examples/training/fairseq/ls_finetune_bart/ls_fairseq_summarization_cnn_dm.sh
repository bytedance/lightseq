#!/usr/bin/env bash
set -ex
THIS_DIR=$(dirname $(readlink -f $0))
cd $THIS_DIR/../../..

if [ ! -d "/tmp/cnn_dm-bin" ]; then
    echo "Downloading dataset"
    wget http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/lightseq/cnn_dm_data/databin_cnn_dm.tar.gz -P /tmp
    tar -xvf /tmp/databin_cnn_dm.tar.gz -C /tmp && rm /tmp/databin_cnn_dm.tar.gz 
fi


if [ ! -d "/tmp/bart.large" ]; then
    echo "Downloading pretrained model"
    wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz -P /tmp
    tar -zxvf /tmp/bart.large.tar.gz -C /tmp && rm /tmp/bart.large.tar.gz
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 lightseq-train /tmp/cnn_dm-bin \
    --restore-file /tmp/bart.large/model.pt \
    --max-tokens 2048 \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch ls_bart_large \
    --criterion ls_label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer ls_adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr 3e-05 --total-num-update 20000 --warmup-updates 500 \
    --fp16 --update-freq 1 \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters
