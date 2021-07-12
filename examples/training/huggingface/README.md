# LightSeq for HuggingFace

This repo contains an example for how to use LightSeq to accerate the training of BERT in HuggingFace [Transformers](https://github.com/huggingface/transformers).

We modify the token classification [examples](https://github.com/huggingface/transformers/tree/master/examples/pytorch/token-classification) in HuggingFace Transformers by replacing their encoder layers with the fused ones in LightSeq.

First you should install these requirements.

```shell
pip install torch ninja transformers seqeval datasets
```

Then you can easily fine-tunes BERT on CoNLL-2003 by running the bash script `run_ner.sh`
or on GLUE by `run_glue.sh`.

The following is our result of GLUE running on single V100-PCIe

| task | config           | result   | speed | speedup |
| ---- | ---------------- | -------- | ----- | ------- |
| MRPC | fp32             | acc 0.87 | 41    | 1       |
| MRPC | fp32 \w lightseq | acc 0.69 | 47    | 1.15    |
| MRPC | fp16             | acc 0.86 | 96    | 2.34    |
| MRPC | fp16 \w lightseq | acc 0.68 | 162   | 3.95    |

Currently, Lightseq use TransformerEncoder in Bert finetune task, and it has a few differences with BertEncoder, which will influence performance of acc or f1. We will support standard BertEncoder ASAP.
