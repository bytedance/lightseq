#!/bin/bash
set -e -u -x

set -ex


mkdir build && cd build

pip3 wheel .. --no-deps -w ../output/