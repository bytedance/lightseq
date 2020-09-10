# LightSeq
LightSeq is a high performance inference library for SOTA NLU/NLG models. It's built on
CUDA official library([cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html),
[Thrust](https://docs.nvidia.com/cuda/thrust/index.html), [CUB](http://nvlabs.github.io/cub/)) and custom kernel functions which are specially fused and
optimized for these widely used models. In addition to model components, we also provide codes
manage model weights trained from deepleanring framework and servers as a custom backend for
[TensorRT Inference
Server](https://docs.nvidia.com/deeplearning/sdk/inference-server-archived/tensorrt_inference_server_120/tensorrt-inference-server-guide/docs/quickstart.html)(referred
to as trtis in the later discussion). With LightSeq, you can easily deploy efficient model services or develop
your own model architectures just with a little code modification.


## Features
- Currently supports Transformer(with beam search) and GPT-2 language model.
- Out-of-the-box end-to-end model server based on trtis.
- In addition to FP32, FP16 inference is also supported with no loss of accuracy even when the model weight is in FP32.
- High inference performance compared with TensorFlow(8x+ speedup on Transformer with beam search,
  4x+ speedup on GPT-2 LM).
- One GPU stream per model, so efficient multi-model on single GPU.


## Performance
We show our experiment on a fr2en translation model which is a Transformer-big with
a beam size of 4 and a target vocabulary size of approximately 30k. The implementation from
[tensor2tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py)
was used as the benchmark of tf-transformer. Due to the lack of tf-beam-search in the fp16
version, we only tested the fp32 version of the tf-transformer for fair comparison.

The following table is a comparison of LightSeq and TensorFlow tested on Tesla P4 and Tesla T4. To
save space, we only show the results of batch_size = 8. More results is available
[here](./docs/performance.md).
<table>
   <tr>
      <td>batch_size</td>
      <td>seq_len</td>
      <td>tf-fp32-p4, ms</td>
      <td>byseq-fp32-p4, ms</td>
      <td>byseq-fp16-t4, ms</td>
      <td>byseq-fp32-p4/tf-fp32-p4, speedup</td>
      <td>byseq-fp16-t4/byseq-fp32-p4, speedup</td>
      <td>byseq-fp16-t4/tf-fp32-p4, speedup</td>
   </tr>
   <tr>
      <td rowspan="8">8</td>
      <td>6</td>
      <td>364</td>
      <td>76</td>
      <td>43</td>
      <td>4.78</td>
      <td>1.77</td>
      <td>8.47</td>
   </tr>
   <tr>
      <td>12</td>
      <td>470</td>
      <td>110</td>
      <td>56</td>
      <td>4.27</td>
      <td>1.96</td>
      <td>8.39</td>
   </tr>
   <tr>
      <td>18</td>
      <td>854</td>
      <td>205</td>
      <td>91</td>
      <td>4.16</td>
      <td>2.25</td>
      <td>9.38</td>
   </tr>
   <tr>
      <td>24</td>
      <td>1381</td>
      <td>318</td>
      <td>139</td>
      <td>4.34</td>
      <td>2.29</td>
      <td>9.94</td>
   </tr>
   <tr>
      <td>36</td>
      <td>1628</td>
      <td>378</td>
      <td>156</td>
      <td>4.3</td>
      <td>2.42</td>
      <td>10.44</td>
   </tr>
   <tr>
      <td>46</td>
      <td>1989</td>
      <td>459</td>
      <td>193</td>
      <td>4.33</td>
      <td>2.38</td>
      <td>10.31</td>
   </tr>
   <tr>
      <td>58</td>
      <td>2683</td>
      <td>617</td>
      <td>254</td>
      <td>4.34</td>
      <td>2.43</td>
      <td>10.56</td>
   </tr>
   <tr>
      <td>70</td>
      <td>4251</td>
      <td>949</td>
      <td>382</td>
      <td>4.47</td>
      <td>2.48</td>
      <td>11.13</td>
   </tr>
</table>

## Code Structure
```python
├── BUILD # bazel build file
├── 3rdparty
│   └── cub-1.8.0 # CUB lib
├── kernels # cuda kernel function
│   ├── common.h  # common function
│   ├── gptKernels.cu.cc # kernel function needed by gpt
│   ├── gptKernels.h
│   ├── transformerKernels.cu.cc # kernel function needed by transformer
│   └── transformerKernels.h
├── model # model infer component
│   ├── decoder.cu.cc # transformer decoder
│   ├── decoder.h 
│   ├── encoder.cu.cc # transformer encoder
│   ├── encoder.h
│   ├── gpt_encoder.cu.cc # gpt
│   └── gpt_encoder.h
├── proto # proto for model weights
│   ├── gpt.proto
│   ├── gpt_weight.cu.cc # model weights loader
│   ├── gpt_weight.h
│   ├── transformer.proto
│   ├── transformer_weight.cu.cc # model weights loader
│   └── transformer_weight.h
├── example # local inference demo
│   ├── gptlm_example.cu.cc # gptlm demo
│   └── transformer_example.cu.cc # transformer demo
├── server # model inference server based on trtis
│   ├── generate_server.cu.cc # transfomer genearate server, multi-target for one source
│   ├── gptlm_server.cu.cc # gptlm server
│   └── transformer_server.cu.cc # transfomer server, one target for one source
└── tools # development tools. e.g. runtime guard, debug
```
## Requirements
- Install Docker and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).
- GPU driver version >= 410.48
- [Login to the NGC registry](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html).

## Quick Start
To avoid problems caused by inconsistent environments, you can use the pre-built trtis container from
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
custom backend of trtis.
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

### Run inference server
To run the end-to-end model server based on trtis, you need to prepare a custom backend [model
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
After starting server, Invoking the [trtis
client](https://docs.nvidia.com/deeplearning/sdk/inference-server-archived/tensorrt_inference_server_120/tensorrt-inference-server-guide/docs/client.html) will get the inference result.

### Serve your own model
In order to serve your own model, you need to [export model](./docs/export_model.md) trained from deeplearning framework(E.g.
TenforFlow, PyTorch) to custom model proto defined by LightSeq. Furthermore, you may need to [build from
source code](./docs/build.md) if you want to modify the model architectures or serve a new model not supported by
LightSeq currently.

## Limitations and Future Plans
LightSeq does not support CPU inference for now and its compilation relies heavily on trtis, we will
try to solve these problems in future. Furthermore, the following will be the focus of our future
work:
- Support more model architectures and decoding search algorithms.
- Int8 inference.
- Device deployment.

## Contact
Any questions or suggestions, please feel free to contact us with wangxiaohui.neo@bytedance.com
