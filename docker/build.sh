#!/bin/bash

set -ex

cd "$(dirname "$0")"
cd ..
docker build . -f docker/Dockerfile.build -t byseqlib/build
docker run --rm -v $(pwd):/workspace byseqlib/build bash /workspace/docker/build_wheel.sh