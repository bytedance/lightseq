# LightSeq for HuggingFace BERT

This repo contains an example for how to use LightSeq to accerate the training of BERT in HuggingFace [Transformers](https://github.com/huggingface/transformers).

We modify the examples like token classification [examples](https://github.com/huggingface/transformers/tree/master/examples/pytorch/token-classification) in HuggingFace Transformers by replacing their encoder layers with the fused ones in LightSeq.

First you should install these requirements.

```shell
pip install torch ninja transformers seqeval datasets
```

Before doing next training, you need to switch to the current directory:
```shell
cd examples/training/huggingface/bert
```

Then you can easily fine-tunes BERT on different tasks by running the bash scripts `task_ner/run_ner.sh`
, `task_glue/run_glue.sh`, `task_qa/run_qa.sh`, etc. From our tests, speedup is about 1.6x.
