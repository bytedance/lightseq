#!/bin/bash

export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}
export PATH=/opt/common_tools:/usr/local/cuda/bin:/usr/local/hdf5:${PATH}

PROJECT_DIR=$(dirname $(readlink -e $0))
cd $PROJECT_DIR

if [[ ! -d "`pwd`/build" ]]; then
    mkdir build
fi

cd build && \
cmake -DCMAKE_BUILD_TYPE=Release -DFP16_MODE=ON -DUSE_NEW_ARCH=ON -DDEBUG_MODE=OFF -DDYNAMIC_API=ON .. && \
make -j${nproc} && \
mkdir $PROJECT_DIR/output && \
mkdir $PROJECT_DIR/output/lib && \
cp $PROJECT_DIR/build/lightseq/csrc/models/liblightseq.so $PROJECT_DIR/output && \
cp -av /usr/local/cuda/lib64/libcublas.so* $PROJECT_DIR/output/lib && \
cp -av /usr/local/cuda/lib64/libcublasLt.so* $PROJECT_DIR/output/lib && \
cp -av /usr/local/cuda/lib64/libcudart.so* $PROJECT_DIR/output/lib && \
cp $PROJECT_DIR/run.sh $PROJECT_DIR/output
