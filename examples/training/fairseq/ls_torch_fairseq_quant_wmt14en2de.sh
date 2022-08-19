#!/usr/bin/env bash
set -ex
THIS_DIR=$(dirname $(readlink -f $0))
cd $THIS_DIR/../../..

if [ ! -d "/tmp/wmt14_en_de" ]; then
    echo "Downloading dataset"
    wget http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/lightseq/wmt_data/databin_wmt14_en_de.tar.gz -P /tmp
    tar -zxvf /tmp/databin_wmt14_en_de.tar.gz -C /tmp && rm /tmp/databin_wmt14_en_de.tar.gz
fi

lightseq-train /tmp/wmt14_en_de/ \
    --task translation \
    --save-dir int4_base1 \
    --finetune-from-model fp16_ende/checkpoint_best.pt \
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
    --fp16 \
    --use-torch-layer \
    --enable-quant \
    --quant-mode qat  \
    --keep-last-epochs 1 --max-epoch 60 \
    --smooth-avg-update 1000 --quant-bits 4 $@ 

# --f-a 1.2 --f-b 0.5