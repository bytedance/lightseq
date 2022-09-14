#! /bin/bash

THIS_DIR=$(dirname $(readlink -f $0))

# You can use multiple NICs in NCCL communication.
# E.g., if every machine has 4 NICs: eth0, eth1, eth2, eth3, you can use the following command.
# export NCCL_SOCKET_IFNAME=eth0,eth1,eth2,eth3

# Set your environment variables according to your training environment,
# for details, please refer to https://pytorch.org/docs/1.10/distributed.html#launch-utility
python3 -m torch.distributed.launch --nproc_per_node=$WORKER_GPU_NUM \
    --nnodes=$WORKER_NUM --node_rank=$WORKER_ID --master_addr=$WORKER_0_HOST \
    --master_port=$WORKER_0_PORT \
    $THIS_DIR/run_gcq_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 1 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm \
    --overwrite_output_dir \
    --fp16 \
    --logging_steps 10 \
    --block_size 512 \
    --module_type 2 \
    --enable_quant false \
    --enable_GCQ true \
    --GCQ_quantile 0.99
