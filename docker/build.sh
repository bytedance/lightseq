#!/bin/bash
set -e -u -x

set -ex

pip3 wheel . --no-deps -w ./output