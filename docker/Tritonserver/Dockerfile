FROM nvcr.io/nvidia/tritonserver:22.01-py3
LABEL maintainer="<zhoubofan@bytedance.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    automake \
    libtool \
    curl \
    make \
    rapidjson-dev \
    g++ \
    unzip \
    sshpass \
    software-properties-common \
    dirmngr\
    apt-transport-https \
    lsb-release \
    ca-certificates \
    openssh-client && \
    rm -rf /var/lib/apt/lists/*

# install HDF5
RUN curl -O -L -C - \
    https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_12_0.tar.gz && \
    tar xf hdf5-1_12_0.tar.gz && \
    rm hdf5-1_12_0.tar.gz && \
    cd hdf5-hdf5-1_12_0 && \
    ./configure --prefix=/usr/local/hdf5 "CFLAGS=-fPIC" "CXXFLAGS=-fPIC" --enable-build-mode=production && \
    make -j$(nproc) && make install && cd .. && rm -rf hdf5-hdf5-1_12_0

RUN curl -O -L -C - \
    https://github.com/protocolbuffers/protobuf/releases/download/v3.13.0/protobuf-cpp-3.13.0.tar.gz && \
    tar xf protobuf-cpp-3.13.0.tar.gz && \
    rm protobuf-cpp-3.13.0.tar.gz && \
    cd protobuf-3.13.0 && ./autogen.sh && \
    ./configure "CFLAGS=-fPIC" "CXXFLAGS=-fPIC" && \
    make -j$(nproc) && make install && ldconfig && cd .. && rm -rf protobuf-3.13.0

# install cmake
ARG CMAKE_PATH=/cmake
RUN mkdir -p ${CMAKE_PATH} && cd ${CMAKE_PATH} && \
    curl -O -L -C - \
    https://github.com/Kitware/CMake/releases/download/v3.18.2/cmake-3.18.2-Linux-x86_64.sh && \
    sh cmake-3.18.2-Linux-x86_64.sh --skip-license && \
    rm cmake-3.18.2-Linux-x86_64.sh && ln -s ${CMAKE_PATH}/bin/cmake /usr/bin/cmake

ENV LD_LIBRARY_PATH /usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/lib:/opt/tritonserver/lib/
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs:/usr/local/lib:/opt/tritonserver/lib/:${LIBRARY_PATH}
ENV PATH /usr/local/cuda/bin:/usr/local/hdf5:/opt/tritonserver/bin:${PATH}

# compile lightseq
RUN cd /opt && git clone --recursive https://github.com/bytedance/lightseq.git && \
    cd lightseq && mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DFP16_MODE=ON -DDEBUG_MODE=OFF -DDYNAMIC_API=ON -DUSE_TRITONBACKEND=ON .. && \
    make -j${nproc}

# organize lightseq-tritonbackend dynamic link library
RUN cd /opt/ && mkdir -p /opt/tritonserver/backends/lightseq && \
    mv /opt/lightseq/build/lightseq/inference/triton_backend/libtriton_lightseq.so /opt/tritonserver/backends/lightseq/ && \
    mv /opt/lightseq/build/lightseq/inference/pywrapper/libliblightseq.so /opt/tritonserver/lib/ && \
    cd /opt/tritonserver && rm -rf /opt/lightseq
