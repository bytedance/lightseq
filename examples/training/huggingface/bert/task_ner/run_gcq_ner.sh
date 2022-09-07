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

# You can use multiple NICs in NCCL communication.
# E.g., if every machine has 4 NICs: eth0, eth1, eth2, eth3, you can use the following command.
# export NCCL_SOCKET_IFNAME=eth0,eth1,eth2,eth3

# Set your environment variables according to your training environment,
# for details, please refer to https://pytorch.org/docs/1.10/distributed.html#launch-utility
python3 -m torch.distributed.launch --nproc_per_node=$WORKER_GPU_NUM \
  --nnodes=$WORKER_NUM --node_rank=$WORKER_ID --master_addr=$WORKER_0_HOST \
  --master_port=$WORKER_0_PORT \
  $THIS_DIR/run_gcq_ner.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name conll2003 \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 16 \
  --num_train_epochs 10 \
  --output_dir /tmp/test-ner \
  --overwrite_output_dir \
  --fp16 \
  --seed 1234 \
  --logging_steps 10 \
  --module_type 2 \
  --enable_quant false \
  --enable_GCQ true \
  --GCQ_quantile 0.99
