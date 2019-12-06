## Build from source code

Byseqlib is built using Docker and trtis containers from
[NVIDIA GPU Cloud (NGC)](https://ngc.nvidia.com/). Before building you must install Docker and
nvidia-docker and login to the NGC registry.

### Build docker image for compilation.
Fistly, you need to build the docker image which is the trtis build environment.
```shell
cur_dir=$(pwd)
git clone https://github.com/NVIDIA/tensorrt-inference-server.git
cd tensorrt-inference-server && git checkout r19.05
docker build -t tensorrtserver_build --target trtserver_build .
```

### Start container
Now you should start container and mount Byseqlib to it.
```shell
cd ${cur_dir}
git clone https://github.com/bytedance/byseqlib.git
cp -r ./byseqlib ./tensorrt-inference-server/src/custom/byseqlib 
docker run --gpus all -it --rm -v ${cur_dir}/tensorrt-inference-server/src:/workspace/src tensorrtserver_build
```

### Build
Finally, build Byseqlib inside container
```shell
# inside container
cd /workspace
# For compatibility with fp16
sed -i '/COMPUTE_CAPABILITIES/s/5.2,6.0,6.1,7.0,7.5/6.0,6.1,7.0,7.5/g' ./.bazelrc
bazel build -c opt src/custom/byseqlib/...
```

