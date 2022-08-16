model=/opt/tiger/begin/lightseq/quant_notvalid_input00_sysv/checkpoint_best.pt


CUDA_VISIBLE_DEVICES=1 lightseq-generate /tmp/wmt14/  \
        --path $model --gen-subset valid --quiet \
        --beam 4 --max-tokens 8192 --remove-bpe --lenpen 0.6 --fp16
