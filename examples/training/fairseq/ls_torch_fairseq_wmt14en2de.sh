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
    --save-dir fp16_ende \
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
    --keep-last-epochs 1
    # --find-unused-parameters \

