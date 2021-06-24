# LightSeq for Fairseq
This repo contains an example for how to use LightSeq to accerate the training of translation task in [Fairseq](https://github.com/pytorch/fairseq).

First you should install these requirements.
```shell
pip install lightseq fairseq sacremoses
```

Then you can train a translation task on wmt14 en2de dataset by running the following script:
```shell
sh examples/training/fairseq/ls_fairseq_wmt14en2de.sh
```

Or you can use LightSeq modules like `--arch ls_transformer_wmt_en_de_big_t2t`,
by adding `--user-dir=${LIGHTSEQ_DIR}/examples/training/fairseq/fs_modules`
to `fairseq-train`.

This script firstly download the dataset and then run native Fairseq
training script using optimized model and optimizer.
The `lightseq-train` command is just a easy-to-use wrapper of `fairseq-train` with adding
LightSeq to `--user-dir`.

LightSeq can achieve about 1.47x speedup using batch size 4096 on 8 V100 GPUs,
compared with original Fairseq implementation. You can delete the `ls` prefix in parameters
to switch to fairseq modules.
