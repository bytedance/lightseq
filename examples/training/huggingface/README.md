# LightSeq for HuggingFace

This repo contains an example for how to use LightSeq to accerate the training of BERT in HuggingFace [Transformers](https://github.com/huggingface/transformers).

We modify the token classification [examples](https://github.com/huggingface/transformers/tree/master/examples/pytorch/token-classification) in HuggingFace Transformers by replacing their encoder layers with the fused ones in LightSeq.

First you should install these requirements.

```shell
pip install torch ninja transformers seqeval datasets
```

Then you can easily fine-tunes BERT on CoNLL-2003 by running the bash script `run_ner.sh`
or on GLUE by `run_glue.sh`.

| task | config           | result  | speed  | speedup |
| ---- | ---------------- | ------- | ------ | ------- |
| ner  | fp32             | f1 0.93 | 75.181 | 1       |
| ner  | fp32 \w lightseq | f1 0.93 | 75.181 | 1       |

Currently, Lightseq use TransformerEncoder in Bert finetune task, and it has a few differences with BertEncoder, which will influence performance of acc or f1. We will support standard BertEncoder ASAP.
