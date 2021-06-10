# LightSeq 2.0 for Fairseq
This repo contains an example for how to use LightSeq 2.0 to accerate the training of translation task in [Fairseq](https://github.com/pytorch/fairseq).

We register a new translation task and adam optimizer using LightSeq 2.0 for Fairseq.

First you should install these requirements.
```shell
pip install torch ninja fairseq
```

Then you can train a translation task on wmt14 en2de dataset by running the following script:
```shell
sh examples/fairseq/ls_fairseq_wmt14en2de.sh
```

This script firstly download the dataset and then run naive Fairseq training script specifing custom task and optimizer.

LightSeq 2.0 can achieve about 1.47x speedup using batch size 4096 on 8 V100 GPUs, compared with original Fairseq implementation.
