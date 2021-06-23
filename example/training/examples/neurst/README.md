# LightSeq for Neurst
This repo contains an example for how to use LightSeq to accerate the training of translation task.

First you should install these requirements.
```shell
pip install subword-nmt pyyaml sacrebleu sacremoses
git clone https://github.com/moses-smt/mosesdecoder.git
```
Then clone Neurst and switch to lightseq branch.
```shell
git clone https://github.com/bytedance/neurst.git
cd neurst/
git checkout lightseq
pip install -e .
```
Install lightseq
```shell
pip install http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/lightseq/tensorflow/lightseq_tf-2.0.1-cp37-cp37m-linux_x86_64.whl
```
Download and preprocess data
```shell
./examples/translation/prepare-wmt14en2de-bpe.sh ../mosesdecoder
```
Traing the model
```shell
python3 -m neurst.cli.run_exp \
    --config_paths wmt14_en_de/training_args.yml,wmt14_en_de/translation_bpe.yml \
    --hparams_set transformer_base \
    --model_dir wmt14_en_de/benchmark_base \
    --enable_xla
```


LightSeq can achieve about 1.33x speedup using batch size 4096 on 8 V100 GPUs,
compared with original tensorflow implementation.
