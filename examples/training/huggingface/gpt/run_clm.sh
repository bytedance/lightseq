#! /bin/bash

THIS_DIR=$(dirname $(readlink -f $0))

# python3 run_clm.py \
python3 -m torch.distributed.launch \
    --nproc_per_node=1 \
    $THIS_DIR/run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm \
    --overwrite_output_dir \
    --fp16 \
    --logging_steps 10 \
    --block_size 512
