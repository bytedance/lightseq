# !/bin/bash
fairseq_path=path/to/model.pt
save_dir=path/to/save_dir

python3 convert_bart_original_pytorch_checkpoint_to_pytorch.py \
    --fairseq_path $fairseq_path \
    --pytorch_dump_folder_path $fairseq_path \
    --hf_config facebook/bart-base --is_bart_base
