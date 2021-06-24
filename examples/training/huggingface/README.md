# LightSeq for HuggingFace
This repo contains an example for how to use LightSeq to accerate the training of BERT in HuggingFace [Transformers](https://github.com/huggingface/transformers).

We modify the token classification [examples](https://github.com/huggingface/transformers/tree/master/examples/pytorch/token-classification) in HuggingFace Transformers by replacing their encoder layers with the fused ones in LightSeq.

First you should install these requirements.
```shell
pip install torch ninja transformers seqeval datasets
```

Then you can easily fine-tunes BERT on CoNLL-2003 by running the bash script `run_ner.sh`.

LightSeq can achieve about 1.3x speedup compared with original HuggingFace Transformers implementation on 8 V100 GPUs.
