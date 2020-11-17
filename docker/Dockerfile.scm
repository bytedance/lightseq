FROM hub.byted.org/base/lab.debian

ENV http_proxy="http://sys-proxy-rd-relay.byted.org:8118"
ENV https_proxy="http://sys-proxy-rd-relay.byted.org:8118"
ENV no_proxy="byted.org"

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
    g++ \
    unzip \
    sshpass \
    openssh-client && \
    curl -fsSL https://mirrors.aliyun.com/nvidia-cuda/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://mirrors.aliyun.com/nvidia-cuda/ubuntu1804/x86_64/ /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-11-0=11.0.221-1 \
    cuda-compat-11-0 \
    cuda-libraries-11-0=11.0.3-1 \
    libnpp-11-0=11.1.0.245-1 \
    cuda-nvtx-11-0=11.0.167-1 \
    libcublas-11-0=11.2.0.252-1 \
    cuda-nvml-dev-11-0=11.0.167-1 \
    cuda-command-line-tools-11-0=11.0.3-1 \
    cuda-nvprof-11-0=11.0.221-1 \
    libnpp-dev-11-0=11.1.0.245-1 \
    cuda-libraries-dev-11-0=11.0.3-1 \
    cuda-minimal-build-11-0=11.0.3-1 \
    libcublas-dev-11-0=11.2.0.252-1 \
    libcusparse-11-0=11.1.1.245-1 \
    libcusparse-dev-11-0=11.1.1.245-1 \
    && ln -s cuda-11.0 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# install protobuf
RUN curl -O -L -C - \
    https://github.com/protocolbuffers/protobuf/releases/download/v3.13.0/protobuf-cpp-3.13.0.tar.gz && \
    tar xf protobuf-cpp-3.13.0.tar.gz && \
    rm protobuf-cpp-3.13.0.tar.gz && \
    cd protobuf-3.13.0 && ./autogen.sh && \
    ./configure "CFLAGS=-fPIC" "CXXFLAGS=-fPIC" && \
    make -j && make install && ldconfig && cd .. && rm -rf protobuf-3.13.0

# install cmake
ARG CMAKE_PATH=/opt/tiger/cmake
RUN mkdir -p ${CMAKE_PATH} && cd ${CMAKE_PATH} && \
    curl -O -L -C - \
    https://github.com/Kitware/CMake/releases/download/v3.18.2/cmake-3.18.2-Linux-x86_64.sh && \
    sh cmake-3.18.2-Linux-x86_64.sh --skip-license && \
    rm cmake-3.18.2-Linux-x86_64.sh && ln -s ${CMAKE_PATH}/bin/cmake /usr/bin/cmake

# install miniconda
ARG CONDA_PATH=/opt/tiger/miniconda
RUN curl -LO -C - https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p ${CONDA_PATH} && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    ${CONDA_PATH}/bin/conda create -y -n py36 python=3.6 && \
    ${CONDA_PATH}/bin/conda create -y -n py37 python=3.7 && \
    ${CONDA_PATH}/bin/conda create -y -n py38 python=3.8

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs:${LIBRARY_PATH}
ENV PATH /opt/common_tools:/usr/local/cuda/bin:${PATH}

CMD ["/sbin/my_init"]