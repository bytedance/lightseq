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
# file tree of model repository
├── <model_repository>/
│  ├── <model_name_1>/            # the directory of model, include parameters and configurations.
│  │  ├── config.pbtxt            # the config of model, more detail is as below.
│  │  ├── <model_file>            # the file of model parameters.
│  │  ├── 1/                      # this empty directory is necessary, which is needed by tritonserver.
│  ├── <model_name_2>/            # ...
│  │  ├── config.pbtxt            # ...
│  │  ├── <model_file>            # ...
│  │  ├── 1/                      # ...
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
  $ sudo docker build -t <docker_image_name> - < Dockerfile
  # Or you can simply pull image which is compiled by ourselves in advance,
  # and you can choose suitable version by replacing `22.01-1` with <tag_name>
  $ sudo docker pull hexisyztem/tritonserver_lightseq:22.01-1
  ```

  - We create a [Dockerfile](https://github.com/bytedance/lightseq/tree/master/examples/triton_backend) ,because lightseq need a dynamic link library which is not contained by nvcr.io/nvidia/tritonserver:22.01-py3.

    If necessary, you can add http_proxy/https_proxy to reduce compile time.

  ```
  # file tree of tritonserver in docker image, user could ignore this part.
  ├── /opt/tritonserver/
  │  ├── backends/                     # the directory of backends, which is used to store backends'
  │  │                                   dynamic link libraries by default.
  │  │  ├── lightseq/                  # the config of model, more detail is as below.
  │  │  │  ├── libtriton_lightseq.so   # the dynamic link library of lightseq's tritonbackend.
  │  │  ├── <other_backends...>        # other directories which are unnecessary for lightseq...
  │  ├── lib/                          # ...
  │  │  ├── libliblightseq.so          # ...
  │  │  ├── libtritonserver.so         # ...
  │  ├── bin/                          # ...
  │  │  ├── tritonserver               # the executable file of tritonserver.
  │  ├── <other_directories...>        # ...
  ```

- Docker Commands:

  ```
  $ sudo docker run --gpus=<num_of_gpus> --rm -p<http_port>:<http_port> -p<grpc_port>:<grpc_port> -v<model_repository>:/models <docker_image_name> tritonserver --model-repository=/models --http-port=<port_id> --grpc-port=<grpc_port>
  ```

  - <num_of_gpus>: int, the number of gpus which are needed by tritonserver.

  - <model_repository>: str, the path of model repository which are organized by yourself.

- Install client requirements:

  ```
  $ pip install tritonclient[all]
  ```

- Run client example:

  ```
  $ export HTTP_PORT=<http_port> && python3 transformer_client_example.py
  ```

## Reference

- [triton-inference-server/backend](https://github.com/triton-inference-server/backend)
- [triton-inference-server/server](https://github.com/triton-inference-server/server)
- [triton-inference-server/client](https://github.com/triton-inference-server/client)
- [triton-inference-server/core](https://github.com/triton-inference-server/core)
- [triton-inference-server/common](https://github.com/triton-inference-server/common)
