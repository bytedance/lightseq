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
  $THIS_DIR/run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --max_seq_length 256 \
  --per_device_train_batch_size 16 \
  --doc_stride 128 \
  --learning_rate 1e-5 \
  --num_train_epochs 16 \
  --output_dir /tmp/quant/squad \
  --overwrite_output_dir \
  --resume_from_checkpoint /tmp/squad/ \
  --fp16 \
  --seed 1234 \
  --logging_steps 10 \
  --module_type 1 \
  --enable_quant true
