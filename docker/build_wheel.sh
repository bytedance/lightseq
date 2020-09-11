#!/bin/bash

set -ex

cd "$(dirname "$0")"
cd ..
rm -rf build/* 
pip3 wheel . --no-deps -w output/wheels