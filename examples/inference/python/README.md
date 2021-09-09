## Examples of exporting models for LightSeq inference

### Switch to the current directory
```shell
cd examples/inference/python
```

### Export models
1. BART
```shell
python export/hf_bart_export.py
```
2. BERT
```shell
python export/hf_bert_export.py
```
3. GPT2
```shell
python export/hf_gpt2_export.py
```
4. Transformer
```shell
python export/ls_fs_transformer_export.py
```

### Inference using LightSeq
1. BART
```shell
python test/ls_bart.py
```
2. BERT
```shell
python test/ls_bert.py
```
3. GPT2
```shell
python test/ls_gpt2.py
```