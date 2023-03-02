# Build from source code

## Requirements
- cudatoolkit-dev >= 10.1 < 11
- protobuf >= 3.13
- cmake >= 3.18

To install cudatoolkit-dev, you could run `conda install -c conda-forge cudatoolkit-dev` or follow the [official guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#runfile), the runfile installation with `--toolkit` arg is recommended. LightSeq can be run on cuda11, but it is not supported to compile with cuda11, cuda11 has build-in cub, it will conflict with submodule cub.

After installation, check the installation of `nvcc` and static libraries (*.a) in `${CUDA_PATH}/lib64`.

To install cmake
```shell
$ curl -O -L -C - https://github.com/Kitware/CMake/releases/download/v3.18.2/cmake-3.18.2-Linux-x86_64.sh
$ sh cmake-3.18.2-Linux-x86_64.sh --skip-license
$ rm cmake-3.18.2-Linux-x86_64.sh && ln -s ${CMAKE_PATH}/bin/cmake /usr/bin/cmake
```

Protobuf need to be built and installed from source.
```shell
$ curl -O -L -C - https://github.com/protocolbuffers/protobuf/releases/download/v3.13.0/protobuf-cpp-3.13.0.tar.gz
$ tar xf protobuf-cpp-3.13.0.tar.gz
$ cd protobuf-3.13.0 && ./autogen.sh
$ ./configure "CFLAGS=-fPIC" "CXXFLAGS=-fPIC"
$ make -j && make install && ldconfig && cd .. && rm -rf protobuf-3.13.0
```
`make install` and `ldconfig` may need to run with `sudo`. If you are encountered with any problem, check [this](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md)

HDF5 also need to be installed.
```shell
$ curl -O -L -C - https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_12_0.tar.gz
$ tar -xzvf hdf5-1_12_0.tar.gz
$ cd hdf5-hdf5-1_12_0
$ ./configure --prefix=/usr/local/hdf5 "CFLAGS=-fPIC" "CXXFLAGS=-fPIC"
$ make
$ make install
$ cd ..
```
If cmake fails with "Could NOT find HDF5", try update `PATH` via `export PATH="$PATH:/usr/local/hdf5"`.

## Build

To build all targets.

```shell
$ mkdir build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DFP16_MODE=ON .. && make -j
```
You can also add -DDEBUG_MODE=ON to output intermediate result for debugging.

To build lightseq wheels.
```shell
$ pip wheel $PROJECT_DIR --no-deps -w $PROJECT_DIR/output/
```

To install python lightseq in development models
```shell
$ PATH=/usr/local/hdf5/:$PATH ENABLE_FP32=1 ENABLE_DEBUG=1 pip3 install -e $PROJECT_DIR
```
