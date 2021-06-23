# LightSeq: A High Performance Library for Sequence Processing and Generation

![logo](https://raw.githubusercontent.com/bytedance/lightseq/master/docs/images/logo.png)

:tada: :tada: :tada: **LightSeq supports fast training for models in the Transformer family now,
please check out [here](https://raw.githubusercontent.com/bytedance/lightseq/master/lightseq/training/README.md) for details. [2021/06/18]**

---

LightSeq is a high performance training and inference library for sequence processing and generation implemented
in CUDA.
It enables highly efficient computation of modern NLP models such as **BERT**, **GPT**,
**Transformer**, etc.
It is therefore best useful for Machine Translation, *Text Generation*, *Dialog*， *Language
Modelling*, *Sentiment analysis*, and other related tasks with sequence data.

The library is built on top of CUDA official
library([cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html),
[Thrust](https://docs.nvidia.com/cuda/thrust/index.html), [CUB](http://nvlabs.github.io/cub/)) and
custom kernel functions which are specially fused and optimized for Transformer model family. In
addition to model components, the inference library also provide easy-to deploy model management and serving backend based on
[TensorRT Inference
Server](https://docs.nvidia.com/deeplearning/sdk/inference-server-archived/tensorrt_inference_server_120/tensorrt-inference-server-guide/docs/quickstart.html).
With LightSeq, one can easily develop modified Transformer architecture with little additional code.

The following is a support matrix of LightSeq **training** library compared with
[DeepSpeed](https://github.com/microsoft/DeepSpeed).

![features](https://raw.githubusercontent.com/bytedance/lightseq/master/lightseq/training/docs/images/features.png)

The following is a support matrix of LightSeq **inference** library compared with
[TurboTransformers](https://github.com/Tencent/TurboTransformers) and
[FasterTransformer](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer).

![support](https://raw.githubusercontent.com/bytedance/lightseq/master/docs/images/support.png)


## Performance

### Training
Here we present the experimental results on WMT14 English to German translation task based on Transformer-big models. We train Transformer models of different sizes on eight NVIDIA Tesla V100/NVIDIA Ampere A100 GPUs with data parallel and fp16 mixed precision.
[Fairseq](https://github.com/pytorch/fairseq) with [Apex](https://github.com/NVIDIA/apex) is choosed as our baseline.

<img src="https://raw.githubusercontent.com/bytedance/lightseq/master/lightseq/training/docs/images/single_step.png"  width="60%" aligned="middle">

We compute speedup on different batch size using the WPS (real words per second) metric.

More results is available [here](https://raw.githubusercontent.com/bytedance/lightseq/master/lightseq/training/docs/performance.md)

### Inference
Here we present the experimental results on neural machine translation based on Transformer-base models using beam search methods.
We choose Tensorflow and
[FasterTransformer](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer) as a comparison.
The implementation from
[tensor2tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py)
was used as the benchmark of Tensorflow.

![nmt](https://raw.githubusercontent.com/bytedance/lightseq/master/docs/images/nmt.png)

More results is available [here](https://raw.githubusercontent.com/bytedance/lightseq/master/docs/performance.md).



## Quick Start

### Training python wrapper
You can use LightSeq operators directly in your codes to build your own models. To simplify the use of individual operators, LightSeq designed a simple and self-contained interface.

For example, if you want to use the encoder layers, you first need to generate a config containing all the arguments of the models and training. Then you can initialize the LightSeq encoder layer using the config and integrate it into you models.

```python
from lightseq.training.ops.pytorch.transformer_encoder_layer import LSTransformerEncoderLayer

config = LSTransformerEncoderLayer.get_config(
    max_batch_tokens=4096,
    max_seq_len=256,
    hidden_size=1024,
    intermediate_size=4096,
    nhead=16,
    attn_prob_dropout_ratio=0.1,
    activation_dropout_ratio=0.1,
    hidden_dropout_ratio=0.1,
    pre_layer_norm=True,
    fp16=True,
    local_rank=0,
)
enc_layer = LSTransformerEncoderLayer(config)
```

Currently, LightSeq supports the separate use of five operations: embedding, encoder layer, decoder layer, criterion and optimizer. You can checkout out the `lightseq/training/ops/pytorch` and `lightseq/training/ops/tensorflow` directory for detail.

### Inference python wrapper

We provide python api to call lightseq, all you need is to install `lightseq` with `pip`, and make sure you have GPU driver not older than 418.40.04.

And check these files `lightseq/inference/proto/*.proto` to prepare your model weights. We provide an example weight file for you to test.

```shell
curl -OL https://github.com/bytedance/lightseq/releases/download/v0.0.1/transformer_weight.tar.gz
tar -zxvf transformer_weight.tar.gz
```

Finally you can run lightseq in only a few lines!

```python
import lightseq.inference as lsi
import numpy as np

test_input = np.array([[5001, 2, 36, 5002]])
transformer = lsi.Transformer("transformer.pb", 32) # 32 is max batch size, it will decide GPU memory occupancy.
result = transformer.infer(test_input)
```

Python api doesn't support GPT for now, and we will get it ready as soon as possible.


## Cite Us

If you use LightSeq in your research, please cite the following paper.

```tex
@InProceedings{wang2021lightseq,
  title = "{L}ight{S}eq: A High Performance Inference Library for Transformers",
    author = "Wang, Xiaohui and Xiong, Ying and Wei, Yang and Wang, Mingxuan and Li, Lei",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies: Industry Papers (NAACL-HLT)",
    month = jun,
    year = "2021",
    publisher = "Association for Computational Linguistics",
    pages = "113--120",
}
```

## Contact

Any questions or suggestions, please feel free to contact us at
wangxiaohui.neo@bytedance.com, xiongying.taka@bytedance.com, weiyang.god@bytedance.com, wangmingxuan.89@bytedance.com, lileilab@bytedance.com
