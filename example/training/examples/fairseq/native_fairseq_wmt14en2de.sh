#!/usr/bin/env bash
set -ex
THIS_DIR=$(dirname $(readlink -f $0))
cd $THIS_DIR/../../

if [ ! -d "wmt14_en_de" ]; then
    wget http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/lightseq/wmt_data/databin_wmt14_en_de.tar.gz
    tar -zxvf databin_wmt14_en_de.tar.gz && rm databin_wmt14_en_de.tar.gz
fi

fairseq-train ./wmt14_en_de/ \
    --arch transformer_wmt_en_de_big_t2t --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8192 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --fp16
