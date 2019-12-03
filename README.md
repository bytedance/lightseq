# trtis_cuda

由于编译依赖众多，所以采用镜像内编译：

1. 构建编译镜像

git clone git@code.byted.org:wangxiaohui.neo/trtis_cuda.git

git clone https://github.com/NVIDIA/tensorrt-inference-server.git

cp -r trtis_cuda tensorrt-inference-server/src/custom/byseqlib 

cd tensorrt-inference-server && git checkout r19.05

docker build -t tensorrtserver_build --target trtserver_build .
(使用众多境外依赖库，可在dockerfile设置http_proxy)

nvidia-docker run -it --rm -v/${path}/${to}/tensorrt-inference-server/src:/workspace/src
tensorrtserver_build （19.03版本docker以前）

或者 docker run --gpus all -it --rm -v/${path}/${to}/tensorrt-inference-server/src:/workspace/src
tensorrtserver_build（19.03版本docker及以后）

2. 在镜像内

cd /workspace

bazel build -c opt src/servers/trtserver

如果要编译fp16版本，需要在镜像内修改下/workspace/.bazelrc的TF_CUDA_COMPUTE_CAPABILITIES变量
改成: 
build --action_env TF_CUDA_COMPUTE_CAPABILITIES="6.0,6.1,7.0,7.5"

bazel build -c opt src/custom/byseqlib/...



## Directory Structure
```
├── 3rdparty
│   └── cub-1.8.0
├── BUILD
├── CONTRIBUTING.md
├── example
│   ├── gptlm_example.cu.cc
│   └── transformer_example.cu.cc
├── kernels
│   ├── common.h
│   ├── gptKernels.cu.cc
│   ├── gptKernels.h
│   ├── transformerKernels.cu.cc
│   └── transformerKernels.h
├── LICENSE
├── model
│   ├── decoder.cu.cc
│   ├── decoder.h
│   ├── encoder.cu.cc
│   ├── encoder.h
│   ├── gpt_encoder.cu.cc
│   └── gpt_encoder.h
├── NOTICE
├── proto
│   ├── gpt.proto
│   ├── gpt_weight.cu.cc
│   ├── gpt_weight.h
│   ├── transformer.proto
│   ├── transformer_weight.cu.cc
│   └── transformer_weight.h
├── README.md
├── server
│   ├── generate_server.cu.cc
│   ├── gptlm_server.cu.cc
│   └── transformer_server.cu.cc
├── test_case
│   ├── case_en2es-5
│   └── case_gpt
└── tools
    ├── util.cu.cc
    └── util.h
```
