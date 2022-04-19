# Tritonbackend Usage

## How To Use Tritonbackend

### How To Compile Tritonbackend

- Execute commands as below

  ```
  $ cd <lightseq_repository>
  $ mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DFP16_MODE=ON -DDEBUG_MODE=OFF -DDYNAMIC_API=ON .. && \
    make -j${nproc}
  ```

   Then you can get outcomes include `libliblightseq.so` and `libtriton_lightseq.so`, Which are needed by model repository.

   You can find libliblightseq.so in this path

  ​     `<lightseq_repository>/build/lightseq/inference/pywrapper/libliblightseq.so`

   While libtriton_lightseq.so is in

  ​      `<lightseq_repository>/build/lightseq/inference/triton_backend/libtriton_lightseq.so`

### How To Organize Model Repository

```
├── <path_to_model_repository>/
│  ├── libliblightseq.so          # dynamic link library of lightseq, which contains the almost
│  │                                implement of lightseq, and should be included by LD_LIBRARY_PATH
│  ├── <model_name_1>/            # the directory of model, include parameters and configurations.
│  │  ├── config.pbtxt            # the config of model, more detail is as below.
│  │  ├── <model_file>            # the file of model parameters.
│  │  ├── 1/                      # this empty directory is necessary, which is needed by tritonserver.
│  │  ├── libtriton_lightseq.so   # dynamic link library of lightseq's tritonbackend
│  ├── <model_name_2>/            # ...
│  │  ├── config.pbtxt            # ...
│  │  ├── <model_file>            # ...
│  │  ├── 1/                      # ...
│  │  ├── libtriton_lightseq.so   # ...
│  ├── #<model_name_vid>...       # more models etc...
```

- The meaning of parameters in config.pbtxt, more information you can find in [Model config of tritonbackend](https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto)

  > ${name}: name of model，**which should be same with <model_name_vid>**
  >
  > ${backend}: **fixed value - "lightseq"**，which is used to recognize the dynamic link library of tritonbackend,  libtriton_lightseq.so
  >
  > ${default_model_filename}: name of model file，**which should be same with <model_file>**
  >
  > ${parameters - value - string_value}: the type of model, which should be supported by lightseq. You can choose `Transformer`|`QuantTransformer`|`Bert`|`Gpt`|`Moe`

- You can see example in [Example Of Triton Model Config](https://github.com/bytedance/lightseq/tree/master/examples/triton_backend/model_repo), while you can also find more detailed information in [Model Config Of Tritonserver](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md).

  - The model files which needed by [Example Of Triton Model Config](https://github.com/bytedance/lightseq/tree/master/examples/triton_backend/model_repo) you can find in [The Guide of Model Export](https://github.com/bytedance/lightseq/blob/master/examples/inference/python/README.md).

### How To Run Tritonserver

#### Run Tritonserver By Docker

- Get tritonserver Docker: [Tritonserver Quickstart](https://github.com/triton-inference-server/server/blob/main/docs/quickstart.md#install-triton-docker-image)

  ```
  $ sudo docker pull nvcr.io/nvidia/tritonserver:22.01-py3
  ```

- Docker Commands:

  ```
  $ sudo docker run --gpus=<num_of_gpus> --rm -e LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/models" -p8000:8000 -p8001:8001 -p8002:8002 -v<model_repository>:/models nvcr.io/nvidia/tritonserver:22.01-py3 tritonserver --model-repository=/models
  ```

  - <num_of_gpus>: int, the number of gpus which are needed by tritonserver.

  - <model_repository>: str, the path of model repository which are organized by yourself.

- Install client requirements:

  ```
  $ pip install tritonclient[all]
  ```

## Reference

- [triton-inference-server/backend](https://github.com/triton-inference-server/backend)
- [triton-inference-server/server](https://github.com/triton-inference-server/server)
- [triton-inference-server/client](https://github.com/triton-inference-server/client)
- [triton-inference-server/core](https://github.com/triton-inference-server/core)
- [triton-inference-server/common](https://github.com/triton-inference-server/common)
