# LightSeq for HuggingFace GPT2

This repo contains an example for how to use LightSeq to accerate the training of GPT2 in HuggingFace [Transformers](https://github.com/huggingface/transformers).

We modify the language modeling [examples](https://github.com/huggingface/transformers/blob/db7d6a80e82d66127b2a44b6e3382969fdc8b207/examples/pytorch/language-modeling/run_clm.py) in HuggingFace Transformers by replacing their encoder layers with LightSeq layers.

First you should install these requirements.

```shell
$ pip install -r requirements.txt
$ bash run_clm.sh
```

Before running the script.make sure your pytorch worksfine with cuda, lightseq doesn't support pytorch cpu mode. You can verify your pytorch on CUDA by the following code.

```python
import torch
x = torch.rand(5, 3).cuda()
print(x)
```

Then you can easily fine-tunes GPT2 on wikitext by running the bash script `run_clm.sh`. From our tests, speedup is about 2x.
