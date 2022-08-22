# Sharded databin for fairseq
This repo contains examples of how to use fairseq with sharded databin to train translation tasks. The goal is to solve the situation where extremely large data cannot be loaded into memory at once.

## Preprocess
You need to use script `ls_preprocess_sharded_databin.sh` to pre-process the data. `--text-dir` is directory path containing the train, valid, test bpe data. `--n-line-per-file` is the number of lines per file, default 100w lines. Other parameters are the same as fairseq-preprocess.
```shell
$ bash ls_preprocess_shared_databin.sh \
    --source-lang src --target-lang tgt \
    --text-dir /path/to/bpe_data \
    --destdir /path/to/databin_dir \
    --n-lines-per-file 1000000
```

## Train
You need to use `ls_translation` task. And add a colon: to the end of the databin path. Databin path can be either the local path, or hdfs path. `--npick` means each epoch picks up n files for training.
```shell
$ lightseq-train hdfs://path/to/hdfs/path: \
    --task ls_translation \
    --source-lang src --target-lang tgt \
    --arch ls_transformer --share-decoder-input-output-embed \
    --optimizer ls_adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --weight-decay 0.0001 \
    --criterion ls_label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8192 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --npick 10 \
    --fp16
```
