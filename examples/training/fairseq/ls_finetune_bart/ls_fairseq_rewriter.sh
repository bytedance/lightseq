#!/usr/bin/env bash
set -ex
THIS_DIR=$(dirname $(readlink -f $0))
cd $THIS_DIR/../../..

PROCESSED_DIR=process_essaylang8merge_bart_base_en
RESTORE_MODEL=path/to/pretrained_model.pt
MODEL_DIR=model_saved
MORE_PARA='--reset-optimizer --reset-dataloader --reset-meters'
SAVE_INTERVAL=1

lightseq-train $PROCESSED_DIR/bin \
     --save-dir $MODEL_DIR \
     --restore-file $RESTORE_MODEL \
     --source-lang src --target-lang trg \
     --layernorm-embedding --share-all-embeddings \
     --arch ls_bart_base \
     --criterion ls_label_smoothed_cross_entropy \
     --label-smoothing 0.1 \
     --max-epoch 2 \
     --max-tokens 4096 \
     --seed 2222 \
     --dropout 0.3 --attention-dropout 0.1 \
     --clip-norm 0.1 \
     --optimizer ls_adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
     --lr 3e-05 --lr-scheduler polynomial_decay --weight-decay 0.01 \
     --fp16 \
     --skip-invalid-size-inputs-valid-test \
     --find-unused-parameters \
     --reset-lr-scheduler \
     --max-source-positions 512 \
     --max-target-positions 512 \
     $MORE_PARA
