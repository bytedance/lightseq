model=/opt/tiger/begin/lightseq/int4/checkpoint_best.pt


CUDA_VISIBLE_DEVICES=1 lightseq-generate /tmp/wmt14/  \
        --path $model/checkpoint_best.pt --gen-subset test --quiet \
        --beam 4 --max-tokens 8192 --remove-bpe --lenpen 0.6 --fp16

CUDA_VISIBLE_DEVICES=1 lightseq-generate /tmp/wmt14/  \
        --path $model/checkpoint_last.pt --gen-subset test --quiet \
        --beam 4 --max-tokens 8192 --remove-bpe --lenpen 0.6 --fp16
