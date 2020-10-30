#!/bin/bash

set -ex

cd "$(dirname "$0")"
docker build . \
  --build-arg http_proxy=sys-proxy-rd-relay.byted.org:8118 \
  --build-arg https_proxy=sys-proxy-rd-relay.byted.org:8118 \
  -f docker/Dockerfile.build \
  -t byseqlib/build
  
docker run --rm -v $(pwd):/workspace byseqlib/build bash /workspace/docker/build_wheels.sh