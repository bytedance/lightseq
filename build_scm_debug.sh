#!/bin/bash
set -e -u -x

PROJECT_DIR=$(dirname $(realpath $0))
cd $PROJECT_DIR

export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}
export PATH=/opt/common_tools:/usr/local/cuda/bin:${PATH}
# Compile wheels
for PYBIN in /opt/tiger/miniconda/envs/py*/bin; do
    "${PYBIN}/pip" wheel $PROJECT_DIR --no-deps -w $PROJECT_DIR/output/ --global-option build_ext --global-option --debug
done
