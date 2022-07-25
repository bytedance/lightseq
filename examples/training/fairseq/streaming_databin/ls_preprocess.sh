# --text-dir: Directory path containing the train valid test
# --n-line-per-file: Number of lines per file, default 100w lines

text=/opt/tiger/begin/wmt14_en_fr
# Other parameters are the same as fairseq-preprocess
bash preprocess_streaming.sh \
    --source-lang en --target-lang fr \
    --text-dir $text --destdir databin --n-line-per-file 1000000
