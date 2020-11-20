#!/bin/bash
set -e -u -x

PROJECT_DIR=$(dirname $(realpath $0))
cd $PROJECT_DIR

export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}
export PATH=/opt/common_tools:/usr/local/cuda/bin:${PATH}
# Compile wheels
for PYBIN in /opt/tiger/miniconda/envs/py*/bin; do
    ENABLE_FP32=0 ENABLE_DEBUG=0 "${PYBIN}/pip" wheel $PROJECT_DIR -v --no-deps -w $PROJECT_DIR/output/
done
mkdir $PROJECT_DIR/output/wheels_fp16
mv $PROJECT_DIR/output/*.whl $PROJECT_DIR/output/wheels_fp16

for PYBIN in /opt/tiger/miniconda/envs/py*/bin; do
    ENABLE_FP32=1 ENABLE_DEBUG=0 "${PYBIN}/pip" wheel $PROJECT_DIR -v --no-deps -w $PROJECT_DIR/output/
done
mkdir $PROJECT_DIR/output/wheels_fp32
mv $PROJECT_DIR/output/*.whl $PROJECT_DIR/output/wheels_fp32

for PYBIN in /opt/tiger/miniconda/envs/py*/bin; do
    ENABLE_FP32=0 ENABLE_DEBUG=1 "${PYBIN}/pip" wheel $PROJECT_DIR -v --no-deps -w $PROJECT_DIR/output/
done
mkdir $PROJECT_DIR/output/wheels_fp16_debug
mv $PROJECT_DIR/output/*.whl $PROJECT_DIR/output/wheels_fp16_debug

for PYBIN in /opt/tiger/miniconda/envs/py*/bin; do
    ENABLE_FP32=0 ENABLE_DEBUG=1 "${PYBIN}/pip" wheel $PROJECT_DIR -v --no-deps -w $PROJECT_DIR/output/
done
mkdir $PROJECT_DIR/output/wheels_fp32_debug
mv $PROJECT_DIR/output/*.whl $PROJECT_DIR/output/wheels_fp32_debug