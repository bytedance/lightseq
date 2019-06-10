# trtis_cuda

由于编译依赖众多，所以采用镜像内编译：

1. 构建编译镜像
https://github.com/NVIDIA/tensorrt-inference-server.git
cp -r $this_repo_dir tensorrt-inference-server/src/custom/transformer 
git checkout r19.05
docker build -t tensorrtserver_build --target trtserver_build .
nvidia-docker run -it --rm -v/path/to/tensorrtserver/src:/workspace/src tensorrtserver_build

2. 在镜像内
cd /workspace
bazel build -c opt src/custom/...
