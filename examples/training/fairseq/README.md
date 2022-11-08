# LightSeq Training for Fairseq

## Table of contents
- [Accerate translation task](#accerate-translation-task)
    - [Train](#train)
    - [Evaluation](#evaluation)
    - [Generate](#generate)
- [Sharded databin for Fairseq](#sharded-databin-for-fairseq)
    - [Preprocess](#preprocess)
    - [Train](#train)
- [GCQ for Fairseq](#gcq-for-fairseq)
- [Fine-tune BART model](#fine-tune-bart-model)

This repo contains examples for how to use LightSeq to accerate the training of [Fairseq](https://github.com/pytorch/fairseq).

First you should install these requirements.
```shell
pip install lightseq fairseq sacremoses
```

## Accerate translation task
### Train
Then you can train a translation task on wmt14 en2de dataset using LightSeq by running the following script:
```shell
sh examples/training/fairseq/ls_fairseq_wmt14en2de.sh
```

Or you can use LightSeq modules like `--arch ls_transformer_wmt_en_de_big_t2t`,
by adding `--user-dir=${LIGHTSEQ_DIR}/lightseq/training/cli/fs_modules`
to `fairseq-train`.

You can use `--use-torch-layer` to replace LightSeq layers with custom Torch layers based on native Fairseq layers.

You can use `--enable-quant` and `--quant-mode qat` to run quantization aware training for subsequent LightSeq fast int8 inference.

This script firstly download the dataset and then run native Fairseq
training script using optimized model and optimizer.
The `lightseq-train` command is just a easy-to-use wrapper of `fairseq-train` with adding
LightSeq to `--user-dir`.

We also provide other training scripts to support custom Torch layers and quantization. All model files have been publicly released. **Refer to [examples/inference/python/README.md](../../../examples/inference/python/README.md) for more training, export and inference details.**

LightSeq can achieve about 1.47x speedup using batch size 4096 on 8 V100 GPUs,
compared with original Fairseq implementation. You can delete the `ls` prefix in parameters
to switch to fairseq modules.

### Evaluation
Then you can evaluate on wmt14 en2de dataset by running the following command:
```shell
lightseq-validate /tmp/wmt14_en_de/ \
    --valid-subset valid \
    --path checkpoints/checkpoint_best.pt \
    --task translation \
    --max-tokens 8192 \
    --criterion ls_label_smoothed_cross_entropy \
    --fp16 \
    --quiet
```

### Generate
You can also generate on wmt14 en2de dataset by running the following command:
```shell
lightseq-generate /tmp/wmt14_en_de/ \
    --gen-subset test \
    --path checkpoints/checkpoint_best.pt \
    --task translation \
    --batch-size 128 \
    --beam 4 \
    --lenpen 0.6 \
    --fp16 \
    --quiet \
    --scoring sacrebleu
```

## Sharded databin for Fairseq
To solve the situation where extremely large data cannot be loaded into memory at once, we can use Fairseq with sharded databin to train translation tasks.

### Preprocess
You need to use script `ls_preprocess_sharded_databin.sh` to pre-process the data. `--text-dir` is directory path containing the train, valid, test bpe data. `--n-line-per-file` is the number of lines per file, default 100w lines. Other parameters are the same as fairseq-preprocess.
```shell
$ bash ls_preprocess_shared_databin.sh \
    --source-lang src --target-lang tgt \
    --text-dir /path/to/bpe_data \
    --destdir /path/to/databin_dir \
    --n-lines-per-file 1000000
```

### Train
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

## GCQ for Fairseq
You can train a translation task on wmt14 en2de dataset using LightSeq with GCQ by running the following script:
```shell
sh ls_fairseq_gcq_wmt14en2de.sh
```

You can use `--enable_GCQ` to enable GCQ in your multi-machine distributed training.
You can set `--GCQ_quantile` to a float value between 0.0 and 1.0, which will use the quantile of gradient bucket as clip-max value when quantizing gradients. E.g., when setting `--GCQ_quantile` 0.99, the clip-max value is equal to the 0.99-th quantile of gradient bucket.

You can use multiple NICs in NCCL communication. E.g., if every machine has 4 NICs: eth0, eth1, eth2, eth3, you can use the following command:
```shell
export NCCL_SOCKET_IFNAME=eth0,eth1,eth2,eth3
```

## Fine-tune BART model
You can fine-tune a BART model using the script `ls_finetune_bart/ls_fairseq_summarization_cnn_dm.sh`, and then convert the model to Hugging Face format using `ls_finetune_bart/convert_lightseq_to_huggingface.sh`.