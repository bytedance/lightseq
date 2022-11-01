# LightSeq: A High Performance Library for Sequence Processing and Generation

![logo](./docs/inference/images/logo.png)

---

## Table Of Contents
- [Release Notes](#release-notes)
- [Introduction](#introduction)
    - [Support Matrix](#support-matrix)
- [Installation](#installation)
    - [Install from PyPI](#install-from-pypi)
    - [Build from Source](#build-from-source)
- [Getting Started](#getting-started)
    - [Fast training from Fairseq](#fast-training-from-fairseq)
    - [Fast inference from Fairseq](#fast-inference-from-fairseq)
    - [Fast inference from Hugging Face BERT](#fast-inference-from-hugging-face-bert)
    - [Fast deployment using inference server](#fast-deployment-using-inference-server)
- [Performance](#performance)
- [Cite Us](#cite-us)
- [Contact](#contact)
- [We are Hiring!](#we-are-hiring)

## Release Notes
**[2022.10.25]** LightSeq release v3.0.0 version, which supports int8 mixed-precision training and inference for Transformer-based models.

**[2021.06.18]** LightSeq release v2.0.0 version, which supports fp16 mixed-precision training for Transformer-based models.

**[2019.12.06]** LightSeq release v1.0.0 version, which supports fp16 mixed-precision inference for Transformer-based models.

## Introduction
LightSeq is a high performance training and inference library for sequence processing and generation implemented in CUDA.
It enables highly efficient computation of modern NLP models such as *BERT*, *GPT*, *Transformer*, etc.
It is therefore best useful for *Machine Translation*, *Text Generation*, *Dialog*, *Language Modeling*, *Sentiment Analysis*, and other related tasks with sequence data.

The library is built on top of CUDA official
library([cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html),
[Thrust](https://docs.nvidia.com/cuda/thrust/index.html), [CUB](http://nvlabs.github.io/cub/)) and
custom kernel functions which are specially fused and optimized for Transformer model family. In
addition to model components, the inference library also provide easy-to deploy model management and serving backend based on
[TensorRT Inference
Server](https://docs.nvidia.com/deeplearning/sdk/inference-server-archived/tensorrt_inference_server_120/tensorrt-inference-server-guide/docs/quickstart.html).
With LightSeq, one can easily develop modified Transformer architecture with little additional code.

LightSeq supports multiple features, including
* training and inference of multiple Transformer-based models, such as Transformer, BERT, GPT2, etc.
* training of multiple modules, such as embedding, Transformer encoder, Transformer decoder, criterion and optimizer.
* multiple decoding methods, such as beam search, diverse beam search and sampling.
* deep integration with multiple training codebases, such as Fairseq, Hugging Face and DeepSpeed.

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

Detailed building introduction is available [here](docs/inference/build.md).

## Getting Started
We provide several samples here to show the usage of LightSeq. Complete user guide is available [here](docs/guide.md).

### Fast training from Fairseq
LightSeq integrates all the fast and lightning modules into Fairseq.

Firstly install the two following requirements:
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

More usage is available [here](./lightseq/training/README.md).

### Fast inference from Fairseq
After training using above scripts, you can fastly infer the models using LightSeq.

You should transform the fp16 PyTorch weights to LightSeq protobuf or HDF5:
```shell
python export/fairseq/ls_fs_transformer_export.py
```

(Optional) You can also transform the int8 PyTorch weights to LightSeq protobuf or HDF5:
```shell
python export/fairseq/ls_fs_quant_transformer_export.py
```

Once obtaining the LightSeq weights, you can fastly infer them using the following code:
```python
import lightseq.inference as lsi
model = lsi.Transformer(MODEL_PATH, MAX_BATCH_SIZE)
results = model.infer([[63, 47, 65, 1507, 88, 74, 10, 2057, 362, 9, 284, 6, 2, 1]])
```
Here MODEL_PATH is the path of your LightSeq weights and MAX_BATCH_SIZE is the maximal batch size of your input sentences.

You can also fastly infer the int8 LightSeq weights by replacing the `lsi.Transformer` with `lsi.QuantTransformer`.

More usage is available [here](./lightseq/inference/README.md).

### Fast inference from Hugging Face BERT
We provide an end2end bert-base example to see how fast Lightseq is compared to original Hugging Face.

First you should install the requirements and locate to the specified directory:
```shell
pip install transformers
cd examples/inference/python
```

Then you can check the performance by simply running following commands. `hf_bert_export.py` is used to transform PyTorch weights to LightSeq protobuf or HDF5.
```shell
python export/huggingface/hf_bert_export.py
python test/ls_bert.py
```

More usage is available [here](./lightseq/inference/README.md).

### Fast deployment using inference server
We provide a docker image which contains tritonserver and LightSeq's dynamic link library, and you can deploy a inference server by simply replacing the model file with your own model file.
```shell
sudo docker pull hexisyztem/tritonserver_lightseq:22.01-1
```

More usage is available [here](./examples/triton_backend/README.md).

## Performance

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
