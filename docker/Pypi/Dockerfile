FROM quay.io/pypa/manylinux2014_x86_64

RUN yum -y install yum-utils && yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo && \
    yum install --setopt=obsoletes=0 -y \
    # base
    cuda-cudart-11-3-11.3.58-1 \
    cuda-compat-11-3 \
    # runtime
    cuda-libraries-11-3-11.3.0-1 \
    cuda-nvtx-11-3-11.3.58-1 \
    libnpp-11-3-11.3.3.44-1 \
    libcublas-11-3-11.4.2.10064-1 \
    libnccl-2.9.6-1+cuda11.3 \
    # devel
    cuda-nvml-devel-11-3-11.3.58-1 \
    cuda-command-line-tools-11-3-11.3.0-1 \
    libcublas-devel-11-3-11.4.2.10064-1 \
    cuda-cudart-devel-11-3-11.3.58-1 \
    cuda-libraries-devel-11-3-11.3.0-1 \
    cuda-minimal-build-11-3-11.3.0-1 \
    cuda-nvprof-11-3-11.3.58-1 \
    libnpp-devel-11-3-11.3.3.44-1 \
    libnccl-devel-2.9.6-1+cuda11.3 \
    && ln -s cuda-11.3 /usr/local/cuda \
    && yum clean all \
    && rm -rf /var/cache/yum/*

RUN yum install -y yum-plugin-versionlock && yum versionlock libcublas11

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

# install protobuf
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

# install HDF5
RUN curl -O -L -C - \
    https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_12_0.tar.gz && \
    tar xf hdf5-1_12_0.tar.gz && \
    rm hdf5-1_12_0.tar.gz && \
    cd hdf5-hdf5-1_12_0 && \
    ./configure --prefix=/usr/local/hdf5 "CFLAGS=-fPIC" "CXXFLAGS=-fPIC" --enable-build-mode=production && \
    make -j$(nproc) && make install && cd .. && rm -rf hdf5-hdf5-1_12_0

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs:${LIBRARY_PATH}
ENV PATH /usr/local/cuda/bin:/usr/local/hdf5:${PATH}
