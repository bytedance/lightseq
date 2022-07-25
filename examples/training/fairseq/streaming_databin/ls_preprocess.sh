#!/usr/bin/env bash
set -ex
THIS_DIR=$(dirname $(readlink -f $0))
cd $THIS_DIR/../../..

# --text-dir: Directory path containing the train valid test
# --n-line-per-file: Number of lines per file, default 100w lines
# Other parameters are the same as fairseq-preprocess
bash preprocess_streaming.sh \
    --source-lang en --target-lang fr \
    --text-dir /path/to/text_dir --destdir databin --n-line-per-file 1000000
