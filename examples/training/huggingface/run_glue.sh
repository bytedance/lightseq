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

export TASK_NAME=mrpc

# v100 fp32 40 samples/sec, 0.87 acc
# v100 fp32 lightseq 47 samples/sec, 0.68 acc
# v100 torch amp 84 samples/sec, 0.86 acc
# v100 torch amp lightseq 163 samples/sec, 0.68 acc
# v100 apex amp 84 samples/sec, 0.86 acc


python3 $THIS_DIR/run_glue.py \
  --model_name_or_path bert-large-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir \
  --with_lightseq true \
  --fp16 \
  --fp16_full_eval \
  --fp16_backend apex \
