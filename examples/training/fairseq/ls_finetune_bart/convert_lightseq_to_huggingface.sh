# !/bin/bash

# The model's directory should contain both source and target vocabulary files
fairseq_path=/path/to/model.pt
save_dir=/path/to/save_dir

python3 convert_lightseq_to_huggingface.py \
    --fairseq_path $fairseq_path \
    --pytorch_dump_folder_path $save_dir \
    --hf_config facebook/bart-base
