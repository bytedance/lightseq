#! /bin/bash

THIS_DIR=$(dirname $(readlink -f $0))

python3 -m torch.distributed.launch \
    --nproc_per_node=1 \
    $THIS_DIR/run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 2 \
    --do_train \
    --do_eval \
    --output_dir /tmp/quant/test-clm \
    --overwrite_output_dir \
    --resume_from_checkpoint /tmp/test-clm \
    --fp16 \
    --logging_steps 10 \
    --block_size 512 \
    --module_type 1 \
    --enable_quant true
