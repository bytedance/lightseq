# LightSeq: A High Performance Library for Sequence Processing and Generation

![logo](./docs/inference/images/logo.png)

---

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

### Build from source
You can also build from source:
```shell
PATH=/usr/local/hdf5/:$PATH ENABLE_FP32=0 ENABLE_DEBUG=0 pip install -e $PROJECT_DIR
```

Detailed building introduction is available [here](docs/inference/build.md).

## Getting Started

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
