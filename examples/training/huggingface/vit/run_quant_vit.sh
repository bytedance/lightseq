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

python3 -m torch.distributed.launch \
  --nproc_per_node=1 \
  $THIS_DIR/run_vit.py \
  --dataset_name beans \
  --output_dir /tmp/quant/beans_outputs \
  --resume_from_checkpoint /tmp/beans_outputs/ \
  --overwrite_output_dir \
  --remove_unused_columns False \
  --do_train \
  --do_eval \
  --learning_rate 2e-6 \
  --num_train_epochs 45 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --logging_steps 10 \
  --seed 1337 \
  --fp16 \
  --module_type 1 \
  --enable_quant true
