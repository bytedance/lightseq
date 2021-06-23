# LightSeq: A High Performance Library for Sequence Processing and Generation

![logo](https://raw.githubusercontent.com/bytedance/lightseq/master/docs/images/logo.png)

:tada: :tada: :tada: **LightSeq supports fast training for models in the Transformer family now,
please check out [here](./lightseq/training/README.md) for details. [2021/06/18]**

[推理模块中文版本介绍](https://bytedance.feishu.cn/docs/doccnUJ5X9WWEdQxXwzbPJ581J0#)

LightSeq is a high performance training and inference library for sequence processing and generation implemented
in CUDA.
It enables highly efficient computation of modern NLP models such as **BERT**, **GPT**,
**Transformer**, etc.
It is therefore best useful for *Machine Translation*, *Text Generation*, *Dialog*， *Language
Modelling*, *Sentiment analysis*, and other related tasks with sequence data.

The library is built on top of CUDA official
library([cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html),
[Thrust](https://docs.nvidia.com/cuda/thrust/index.html), [CUB](http://nvlabs.github.io/cub/)) and
custom kernel functions which are specially fused and optimized for Transformer model family. In
addition to model components, the library also provide easy-to deploy model management and serving backend based on
[TensorRT Inference
Server](https://docs.nvidia.com/deeplearning/sdk/inference-server-archived/tensorrt_inference_server_120/tensorrt-inference-server-guide/docs/quickstart.html)(referred
to as TRTIS in the later discussion).
With LightSeq, one can easily develop modified Transformer architecture with little additional code.

## Features

- Comprehensive sequence modeling support, including Bert, GPT, Transformer and their VAE variants.
- Various search methods, such as beam search, diverse beam search, topp/topk sampling.
- Out-of-the-box rich middlewares for model service based on TRTIS, such as dynamic batch,
  multi-model on single GPU.
- Lightening fast training speed for supported models.
- Lightening fast inference performance compared with Deeplearning framework and other inference
  libraries.

The following is a support matrix of LightSeq compared with
[TurboTransformers](https://github.com/Tencent/TurboTransformers) and
[FasterTransformer](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer).

![support](https://raw.githubusercontent.com/bytedance/lightseq/master/docs/images/support.png)

## Code Structure
At present, the code about **training** is in directory `./lightseq/training`, and other directories are all about **inference**.
The training and inference parts are currently separate, and we will merge them in the near future.

## Performance

Here are the experimental results on neural machine translation and text generation.
The models of these two tasks are Transformer-base, but use beam search and sampling search methods
respectively.
We choose Tensorflow and
[FasterTransformer](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer) as a comparison.
The implementation from
[tensor2tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py)
was used as the benchmark of Tensorflow.

More results is available [here](./docs/performance.md).

- Neural machine translation
![nmt](https://raw.githubusercontent.com/bytedance/lightseq/master/docs/images/nmt.png)

- Text generation
![generation](https://raw.githubusercontent.com/bytedance/lightseq/master/docs/images/generation.png)


## Quick Start

### Fast training from Fairseq

You can experience lightning fast training by running following commands,
Firstly install these requirements.

```shell
pip install lightseq fairseq sacremoses
```

Then you can train a translation task on wmt14 en2de dataset by running the following script

```shell
sh lightseq/training/examples/fairseq/ls_fairseq_wmt14en2de.sh
```

To compare lightseq with fairseq, delete the arguments with `ls_`prefix to using the original fairseq implementation

### Fast inference from HuggingFace bart

We provide an end2end bart-base example to see how fast Lightseq is compared to HuggingFace. First you should install these requirements.

```shell
pip install torch tensorflow transformers lightseq
cd example/python
```

then you can check the performance by simply running following commands. `hf_bart_export.py` is used to transform pytorch weights to LightSeq protobuffer.

```shell
python hf_bart_export.py
python ls_bart.py
```

on our Tesla V100 we can get following output, 10x speedup have been obtained from running LightSeq rather than HuggingFace.

```log
=========lightseq=========
lightseq generating...
lightseq time: 0.03398153930902481s
lightseq results:
I love that girl, but she does not love me.
She is so beautiful that I can not help glance at her.
Nothing's gonna change my love for you.
Drop everything now. Meet me in the pouring rain. Kiss me on the sidewalk.
=========huggingface=========
huggingface generating...
huggingface time: 0.3320543058216572s
huggingface results:
I love that girl, but she does not love me.
She is so beautiful that I can not help glance at her.
Nothing's gonna change my love for you.
Drop everything now. Meet me in the pouring rain. Kiss me on the sidewalk.
```

LightSeq installation from pypi only supports python 3.6 to 3.8 on Linux for now. Consider compiling from source if you have other environments.

### Inference python wrapper

We provide python api to call lightseq, all you need is to install `lightseq` with `pip`, and make sure you have GPU driver not older than 418.40.04.

And check these files `proto/*.proto` to prepare your model weights. We provide an example weight file for you to test.

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

### Run inference server

#### Requirements

- Install Docker and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).
- GPU driver version >= 410.48
- [Login to the NGC registry](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html).

To avoid problems caused by inconsistent environments, you can use the pre-built TRTIS container from
[NVIDIA GPU Cloud (NGC)](https://ngc.nvidia.com/). To start the given container, you need to install
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker) and make your GPU driver version >= 410.48

```shell
docker pull nvcr.io/nvidia/tensorrtserver:19.05-py3
#
docker run --gpus '"device=0"' -it --rm -p8000:8000 -p8001:8001 -p8002:8002 -v
/${current}/${path}:/quick_start nvcr.io/nvidia/tensorrtserver:19.05-py3 /bin/bash
# inside container
cd /quick_start
```

### Use our pre-build lib

To quickly deploy your model that supported by LightSeq currently, you can download the pre-built libraries
from the GitHub release page corresponding to the release version you are interested in. In each release
version, we will upload binary executable example and dynamic link library of models which is a
custom backend of TRTIS.

```shell
wget https://github.com/bytedance/lightseq/releases/download/${VERSION}/${VERSION}_libs.tar.gz
tar -zxvf ${VERSION}_libs.tar.gz
```

### Run local inference demo

To run local inference demo, you need to prepare model weights saved in custom proto defined by
LightSeq and input token ids. We provide a GPT-LM model and its corresponding input token ids:

```shell
wget https://github.com/bytedance/lightseq/releases/download/v0.0.1/v0.0.1_gptlm.pkg.tar.gz
tar -zxvf v0.0.1_gptlm.pkg.tar.gz
# fp32 example
./{VERSION}_libs/gptlm_example.fp32 ./v0.0.1_gptlm.pkg/gpt.pb ./v0.0.1_gptlm.pkg/test_case
# fp16 example
./{VERSION}_libs/gptlm_example.fp16 ./v0.0.1_gptlm.pkg/gpt.pb ./v0.0.1_gptlm.pkg/test_case
```

To run the end-to-end model server based on TRTIS, you need to prepare a custom backend [model
repository](https://docs.nvidia.com/deeplearning/sdk/inference-server-archived/tensorrt_inference_server_120/tensorrt-inference-server-guide/docs/model_repository.html#custom-backends) like this:

```shell
models/
  <model-name>/
    config.pbtxt # configuration
    xxx # model weights
    1/
      libyyy.so # custom dynamic link library
```

With the pre-built libraries and example weights mentioned above, you can easily run a server:

```shell
mkdir -p ./model_zoo/gptlm/1
wget https://github.com/bytedance/lightseq/releases/download/v0.0.1/v0.0.1_gptlm.config.pbtxt
mv v0.0.1_gptlm.config.pbtxt model_zoo/gptlm/config.pbtxt
cp ./v0.0.1_gptlm.pkg/gpt.pb model_zoo/gptlm/gpt.pb
cp ./{VERSION}_libs/libgptlm.so.fp32 model_zoo/gptlm/1/libgptlm.so
# or fp16 server
# cp ./{VERSION}_libs/libgptlm.so.fp16 model_zoo/gptlm/1/libgptlm.so
export MODEL_ZOO="/quick_start/model_zoo"
trtserver --model-store=${MODEL_ZOO}
```

After starting server, Invoking the [TRTIS
client](https://docs.nvidia.com/deeplearning/sdk/inference-server-archived/tensorrt_inference_server_120/tensorrt-inference-server-guide/docs/client.html) will get the inference result.

### Serve your own model

In order to serve your own model, you need to [export model](./docs/export_model.md) trained from deeplearning framework(E.g.
TenforFlow, PyTorch) to custom model proto defined by LightSeq. Furthermore, you may need to [build from
source code](./docs/build.md) if you want to modify the model architectures or serve a new model not supported by
LightSeq currently.

## Limitations and Future Plans

LightSeq does not support CPU inference for now and its compilation relies heavily on TRTIS, we will
try to solve these problems in future. Furthermore, the following will be the focus of our future
work:

- Support more model architectures and decoding search algorithms.
- Int8 inference.
- Device deployment.

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
