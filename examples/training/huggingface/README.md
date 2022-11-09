# LightSeq Training for Hugging Face

## Table of Contents
- [Training of Hugging Face Models](#training-of-hugging-face-models)
    - [BERT](#bert)
    - [GPT2](#gpt2)
    - [BART](#bart)
    - [ViT](#vit)
- [Gradient Communication Quantization](#gradient-communication-quantization)

## Training of Hugging Face Models
We provide multiple examples of using LightSeq to accelerate Hugging Face model training.

### BERT
Before doing next training, you need to switch to the BERT directory:
```shell
cd examples/training/huggingface/bert
```

First you should install these requirements:
```shell
pip install torch ninja transformers seqeval datasets
```

Then you can easily fine-tunes BERT on different tasks by running the bash scripts `task_ner/run_ner.sh`
, `task_glue/run_glue.sh`, `task_qa/run_qa.sh`, etc.

You can also fine-tune the models using int8 mixed-precision by running `task_ner/run_quant_ner.sh`.

### GPT2
Before doing next training, you need to switch to the GPT2 directory:
```shell
cd examples/training/huggingface/gpt
```

First you should install these requirements:

```shell
pip install -r requirements.txt
```

Then you can easily fine-tunes GPT2 by running the bash scripts `run_clm.sh`.

You can also fine-tune the models using int8 mixed-precision by running `run_quant_clm.sh`.

### BART
Before doing next training, you need to switch to the GPT2 directory:
```shell
cd examples/training/huggingface/bart/summarization
```

First you should install these requirements:

```shell
pip install -r requirements.txt
```

Then you can easily fine-tunes BART by running the bash scripts `run_summarization.sh`.

### ViT
Before doing next training, you need to switch to the ViT directory:
```shell
cd examples/training/huggingface/vit
```

First you should install these requirements:
```shell
pip install torch ninja transformers seqeval datasets
```

Then you can easily fine-tunes ViT by running the bash scripts `run_vit.sh`.

You can also fine-tune the models using int8 mixed-precision by running `run_quant_vit.sh`.

## Gradient Communication Quantization
LightSeq support Hugging Face training using GCQ. Taking BERT as an example, first you need to switch to BERT directory you can easily fine-tunes BERT with GCQ on different tasks by running the bash scripts `task_ner/run_gcq_ner.sh` , `task_glue/run_gcq_glue.sh`, `task_qa/run_gcq_qa.sh`, etc.

You can use `--enable_GCQ` to enable GCQ in your multi-machine distributed training.
You can set `--GCQ_quantile` to a float value between 0.0 and 1.0, which will use the quantile of  gradient bucket as clip-max value when quantizing gradients. E.g., when setting `--GCQ_quantile` 0.99, the clip-max value is equal to the 0.99-th quantile of gradient bucket.

You can use multiple NICs in NCCL communication. E.g., if every machine has 4 NICs: eth0, eth1, eth2, eth3, you can use the following command.
```shell
export NCCL_SOCKET_IFNAME=eth0,eth1,eth2,eth3
```
