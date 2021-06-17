# LightSeq for Fairseq+DeepSpeed
This repo contains an example for how to use LightSeq to accerate the training of translation task in [Fairseq](https://github.com/pytorch/fairseq), together with [DeepSpeed](https://github.com/microsoft/DeepSpeed) for distributed strategies and optimizers. We provide a new trainer for translation task to connect Fairseq and DeepSpeed.

First you should install these requirements.
```shell
pip install torch ninja fairseq deepspeed
```

Then you can train a translation task on wmt14 en2de dataset by running the following script:
```shell
sh lightseq/training/examples/deepspeed/ds_fairseq_wmt14en2de.sh
```

This script firstly download the dataset, and then run native Fairseq training script using DeepSpeed launcher without any other parameter modifications.
