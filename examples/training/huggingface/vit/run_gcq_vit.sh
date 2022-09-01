# Copyright 2021 The LightSeq Team
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

THIS_DIR=$(dirname $(readlink -f $0))

# Set your environment variables according to your training environment,
# for details, please refer to https://pytorch.org/docs/1.10/distributed.html#launch-utility
python3 -m torch.distributed.launch --nproc_per_node=$WORKER_GPU_NUM \
    --nnodes=$WORKER_NUM --node_rank=$WORKER_ID --master_addr=$WORKER_0_HOST \
    --master_port=$WORKER_0_PORT \
    $THIS_DIR/run_gcq_vit.py \
    --dataset_name beans \
    --output_dir /tmp/beans_outputs \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337 \
    --fp16 \
    --with_lightseq true \
    --enable_GCQ true \
    --GCQ_quantile 0.99 
