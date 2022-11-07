# LightSeq: A High Performance Library for Sequence Processing and Generation

![logo](./docs/images/logo.png)

---

## Table Of Contents
- [Release Notes](#release-notes)
- [Introduction](#introduction)
    - [Support Matrix](#support-matrix)
- [Performance](#performance)
    - [Speedup of Transformer Training](#speedup-of-transformer-training)
    - [Speedup of BERT Training](#speedup-of-bert-training)
    - [Speedup of Transformer Inference](#speedup-of-transformer-inference)
    - [Speedup of BERT Inference](#speedup-of-bert-inference)
- [Installation](#installation)
    - [Install from PyPI](#install-from-pypi)
    - [Build from Source](#build-from-source)
- [Getting Started](#getting-started)
    - [LightSeq Training from Scratch](#lightseq-training-from-scratch)
    - [LightSeq Training from Fairseq](#lightseq-training-from-fairseq)
    - [LightSeq Training from Hugging Face BERT](#lightseq-training-from-hugging-face-bert)
    - [LightSeq Inference from Fairseq](#lightseq-inference-from-fairseq)
    - [LightSeq Inference from Hugging Face BERT](#lightseq-inference-from-hugging-face-bert)
    - [LightSeq Deployment Using Inference Server](#lightseq-deployment-using-inference-server)
- [Cite Us](#cite-us)
- [Contact](#contact)
- [We are Hiring!](#we-are-hiring)

## Release Notes
**[2022.10.25]** Release v3.0.0 version, which supports int8 mixed-precision training and inference. [[中文介绍](https://bytedance.feishu.cn/docx/doxcnZloQZmLgAVU7z1QFlcRPuO)]

**[2021.06.18]** Release v2.0.0 version, which supports fp16 mixed-precision training. [[中文介绍](https://bytedance.feishu.cn/docs/doccn9w7UdOYcEOD99FjFVpdFzf)]

**[2019.12.06]** Release v1.0.0 version, which supports fp16 mixed-precision inference. [[中文介绍](https://bytedance.feishu.cn/docs/doccnUJ5X9WWEdQxXwzbPJ581J0)]

## Introduction
LightSeq is a high performance training and inference library for sequence processing and generation implemented in CUDA.
It enables highly efficient computation of modern NLP and CV models such as BERT, GPT, Transformer, etc.
It is therefore best useful for machine translation, text generation, image classification, and other sequence related tasks.

The library is built on top of CUDA official
library([cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html),
[Thrust](https://docs.nvidia.com/cuda/thrust/index.html), [CUB](http://nvlabs.github.io/cub/)) and
custom kernel functions which are specially fused and optimized for Transformer model family. In
addition to model components, the inference library also provide easy-to-deploy model management and serving backend based on
[TensorRT Inference
Server](https://docs.nvidia.com/deeplearning/sdk/inference-server-archived/tensorrt_inference_server_120/tensorrt-inference-server-guide/docs/quickstart.html).
With LightSeq, one can easily develop modified Transformer architecture with little additional code.

LightSeq supports multiple features, including
* training and inference with fp32, fp16 and int8 precision.
* training and inference of multiple Transformer-based models, such as Transformer, BERT, GPT2, etc.
* training of multiple modules, such as embedding, Transformer encoder, Transformer decoder, criterion and optimizer.
* multiple decoding methods, such as beam search, diverse beam search and sampling.
* deep integration with multiple training codebases, such as Fairseq, Hugging Face and DeepSpeed.

LightSeq training and inference is very fast. Below is the overall performance:
* LightSeq fp16 training achieves a speedup of up to **3x**, compared to PyTorch fp16 training.
* LightSeq int8 training achieves a speedup of up to **5x**, compared to PyTorch QAT (i.e., quantization aware training).
* LightSeq fp16 and int8 inference achieve a speedup of up to **12x** and **15x**, compared to PyTorch fp16 inference, respectively.

### Support Matrix
|    Models    | fp16 Training | fp16 Inference | int8 Training | int8 Inference |
| ------------ | ------------- | -------------- | ------------- | -------------- |
| Transformer  | Yes           | Yes            | Yes           | Yes            |
| BERT         | Yes           | Yes            | Yes           | Yes            |
| GPT2         | Yes           | Yes            | Yes           | Yes            |
| BART         | Yes           | Yes            | -             | -              |
| T5           | -             | Yes            | -             | -              |
| MT5          | -             | Yes            | -             | -              |
| XGLM         | -             | Yes            | -             | -              |
| ViT          | Yes           | Yes            | Yes           | Yes            |
| VAE          | -             | Yes            | -             | -              |
| Multilingual | -             | Yes            | -             | Yes            |

## Performance
We test the speedup of LightSeq training and inference using both fp16 and int8 mix-precision on Transformer and BERT models. The baseline is PyTorch fp16 mix-precision. Training experiments are tested on one A100 GPU and inference experiments are tested on eight A100 GPUs.

More performance results are available [here](./docs/performance).

### Speedup of Transformer Training
| batch token size | PyTorch QAT | LightSeq fp16 | LightSeq int8 |
| ---------------- | ----------- | ------------- | ------------- |
| 512              | 0.36        | 1.99          | 1.86          |
| 1024             | 0.37        | 1.78          | 1.69          |
| 2048             | 0.37        | 1.56          | 1.50          |
| 4096             | 0.39        | 1.47          | 1.44          |
| 8192             | 0.41        | 1.44          | 1.44          |
| 15000            | 0.43        | 1.44          | 1.44          |

### Speedup of BERT Training
| batch token size | PyTorch QAT | LightSeq fp16 | LightSeq int8 |
| ---------------- | ----------- | ------------- | ------------- |
| 8                | 0.45        | 2.12          | 1.99          |
| 16               | 0.44        | 1.92          | 1.80          |
| 32               | 0.42        | 1.59          | 1.52          |
| 64               | 0.46        | 1.62          | 1.58          |
| 128              | 0.46        | 1.74          | 1.70          |
| 256              | 0.46        | 1.68          | 1.73          |

### Speedup of Transformer Inference
| batch size | sequence length | LightSeq fp16 | LightSeq int8 |
|------------|-----------------|---------------|---------------|
| 1          | 8               | 8.00          | 9.33          |
| 1          | 32              | 6.48          | 7.38          |
| 1          | 128             | 6.24          | 6.19          |
| 8          | 8               | 9.38          | 10.71         |
| 8          | 32              | 8.24          | 8.75          |
| 8          | 128             | 6.83          | 7.28          |
| 32         | 8               | 11.82         | 14.44         |
| 32         | 32              | 9.68          | 11.15         |
| 32         | 128             | 6.68          | 7.74          |

### Speedup of BERT Inference
| batch size | sequence length | LightSeq fp16 | LightSeq int8 |
| ---------- | --------------- | ------------- | ------------- |
| 1          | 8               | 9.22          | 9.87          |
| 1          | 32              | 10.51         | 11.30         |
| 1          | 128             | 9.96          | 10.85         |
| 8          | 8               | 9.88          | 10.33         |
| 8          | 32              | 7.79          | 8.22          |
| 8          | 128             | 4.04          | 4.35          |
| 32         | 8               | 10.60         | 11.02         |
| 32         | 32              | 8.11          | 8.85          |
| 32         | 128             | 1.82          | 2.04          |

## Installation
### Install from PyPI
You can install LightSeq from PyPI, which only supports Python 3.6 to 3.8 on Linux:
```shell
pip install lightseq
```

### Build from Source
You can also build from source:
```shell
PATH=/usr/local/hdf5/:$PATH ENABLE_FP32=0 ENABLE_DEBUG=0 pip install -e $PROJECT_DIR
```

Detailed building introduction is available [here](docs/build.md).

## Getting Started
We provide several samples here to show the usage of LightSeq. Refer to the complete [user guide](./docs/guide.md) and [examples](./docs/examples.md) for more details.

### LightSeq Training from Scratch
You can use the modules provided by LightSeq to build your own models. The following is an example of building a Transformer encoder layer.

First, import LightSeq Transformer encoder module:
```python
from lightseq.training import LSTransformerEncoderLayer
```

Then create an encoder configuration, and create a LightSeq Transformer encoder layer initialized with the configuration:
```python
config = LSTransformerEncoderLayer.get_config(
    max_batch_tokens=4096,
    max_seq_len=512,
    hidden_size=1024,
    intermediate_size=4096,
    nhead=16,
    attn_prob_dropout_ratio=0.1,
    activation_dropout_ratio=0.1,
    hidden_dropout_ratio=0.1,
    pre_layer_norm=True,
    activation_fn="relu",
    fp16=True,
    local_rank=0,
)
layer = LSTransformerEncoderLayer(config)
```

In addition to encoder layers, the other modules can be created using similar methods, and then be trained as normal PyTorch models.

More usage is available [here](./examples/training/custom/README.md).

### LightSeq Training from Fairseq
LightSeq integrates all the fast and lightning modules into Fairseq.

First install the two following requirements:
```shell
pip install fairseq==0.10.2 sacremoses
```

You can train a fp16 mix-precision translation task on wmt14 en2de dataset by:
```shell
sh examples/training/fairseq/ls_fairseq_wmt14en2de.sh
```

(Optional) Then you can start int8 mix-precision training on the basis of fp16 pre-training models by:
```shell
sh examples/training/fairseq/ls_fairseq_quant_wmt14en2de.sh
```

More usage is available [here](./examples/training/fairseq/README.md).

### LightSeq Training from Hugging Face BERT
LightSeq replaces the encoder layers of Hugging Face BERT with LightSeq fast layers.

First you should install these requirements:

```shell
pip install transformers seqeval datasets
```

Before doing next training, you need to switch to the following directory:
```shell
cd examples/training/huggingface/bert
```

Then you can easily fine-tune BERT for different tasks. Taking named entity recognition task as an example, you can train the BERT with fp16 mixed-precision using:
```shell
python task_ner/run_ner.sh
```

(Optional) You can also start int8 mix-precision training on the basis of fp16 pre-training models by:
```shell
python task_ner/run_quant_ner.sh
```

More usage is available [here](./examples/training/huggingface/README.md).

### LightSeq Inference from Fairseq
After training using the above scripts, you can quickly infer the models using LightSeq.

You should transform the fp16 PyTorch weights to LightSeq protobuf or HDF5:
```shell
python export/fairseq/ls_fs_transformer_export.py
```

(Optional) You can also transform the int8 PyTorch weights to LightSeq protobuf or HDF5:
```shell
python export/fairseq/ls_fs_quant_transformer_export.py
```

Once obtaining the LightSeq weights, you can quickly infer them using the following code:
```python
import lightseq.inference as lsi
model = lsi.Transformer(MODEL_PATH, MAX_BATCH_SIZE)
results = model.infer([[63, 47, 65, 1507, 88, 74, 10, 2057, 362, 9, 284, 6, 2, 1]])
```
Here MODEL_PATH is the path of your LightSeq weights and MAX_BATCH_SIZE is the maximal batch size of your input sentences.

You can also quickly infer the int8 LightSeq weights by replacing the `lsi.Transformer` with `lsi.QuantTransformer`.

More usage is available [here](./examples/inference/python/README.md).

### LightSeq Inference from Hugging Face BERT
We provide an end2end bert-base example to see how fast Lightseq is compared to original Hugging Face.

First you should install the requirements and locate to the specified directory:
```shell
pip install transformers
cd examples/inference/python
```

Then you can check the performance by simply running the following commands. `hf_bert_export.py` is used to transform PyTorch weights to LightSeq protobuf or HDF5.
```shell
python export/huggingface/hf_bert_export.py
python test/ls_bert.py
```

More usage is available [here](./examples/inference/python/README.md).

### LightSeq Deployment Using Inference Server
We provide a docker image which contains tritonserver and LightSeq's dynamic link library, and you can deploy an inference server by simply replacing the model file with your own model file.
```shell
sudo docker pull hexisyztem/tritonserver_lightseq:22.01-1
```

More usage is available [here](./examples/triton_backend/README.md).

## Cite Us
If you use LightSeq in your research, please cite the following papers.

```
@InProceedings{wang2021lightseq,
    title = "{L}ight{S}eq: A High Performance Inference Library for Transformers",
    author = "Wang, Xiaohui and Xiong, Ying and Wei, Yang and Wang, Mingxuan and Li, Lei",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies: Industry Papers (NAACL-HLT)",
    month = jun,
    year = "2021",
    publisher = "Association for Computational Linguistics",
    pages = "113--120",
}

@article{wang2021lightseq2,
  title={LightSeq2: Accelerated Training for Transformer-based Models on GPUs},
  author={Wang, Xiaohui and Xiong, Ying and Qian, Xian and Wei, Yang and Li, Lei and Wang, Mingxuan},
  journal={arXiv preprint arXiv:2110.05722},
  year={2021}
}
```

## Contact
Any questions or suggestions, please feel free to contact us at
wangxiaohui.neo@bytedance.com, xiongying.taka@bytedance.com, weiyang.god@bytedance.com, zhangzhexi@bytedance.com, zhoubofan@bytedance.com, qian.xian@bytedance.com, wangmingxuan.89@bytedance.com, lilei@cs.ucsb.edu

## We are Hiring!
The LightSeq team is hiring Interns and FTEs with backgrounds in *deep learning system, natural language processing, computer vision, speech, etc*.
We are based in Beijing and Shanghai. If you are interested, please send your resume to wangxiaohui.neo@bytedance.com.
