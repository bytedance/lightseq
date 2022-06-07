#!/usr/bin/env bash

until [[ -z "$1" ]]
do
    case $1 in
        -m)
            shift; MODEL=$1;
            shift;;
        *)
            shift;;
    esac
done

lightseq-infer /tmp/wmt14_en_de/ \
    --gen-subset test \
    --path ${MODEL} \
    --task translation \
    --batch-size 128 \
    --beam 4 \
    --lenpen 0.6 \
    --fp16 \
    --quiet \
    --scoring sacrebleu
