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

export TASK_NAME=sst2

python3 -m torch.distributed.launch \
  --nproc_per_node=1 \
  $THIS_DIR/run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-6 \
  --num_train_epochs 20 \
  --output_dir /tmp/quant/$TASK_NAME/ \
  --overwrite_output_dir \
  --resume_from_checkpoint /tmp/$TASK_NAME/ \
  --fp16 \
  --seed 1234 \
  --logging_steps 10 \
  --module_type 2 \
  --enable_quant true
