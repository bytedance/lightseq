## Examples of exporting models for LightSeq inference

### Switch to the current directory
```shell
cd examples/inference/python
```

### Export models
1. Hugging Face BART

Export Hugging Face BART models to protobuf/hdf5 format.
```shell
python export/huggingface/hf_bart_export.py
```
2. Hugging Face BERT

Export Hugging Face BERT models to hdf5 format.
```shell
python export/huggingface/hf_bert_export.py
```
3. Hugging Face GPT2

Export Hugging Face GPT2 models to hdf5 format.
```shell
python export/huggingface/hf_gpt2_export.py
```
4. Fairseq Transformer using LightSeq training library

Export Fairseq Transformer models training with LightSeq to protobuf/hdf5 format. Refer to the `examples/training/fairseq` directory for more training details.
```shell
python export/fairseq/ls_fs_transformer_export.py
```
5. Fairseq Transformer using LightSeq training library with int8 quantization

Export Fairseq Transformer models training with LightSeq to protobuf format, and then using int8 quantization to speedup inference. Refer to the `examples/training/fairseq` directory for more training details.
```shell
python export/fairseq/ls_fs_transformer_ptq_export.py
```
**You can compare the speeds between fp16 and int8 inference using above 4th and 5th examples.**

6. LightSeq Transformer

Export LightSeq Transformer models to protobuf/hdf5 format. Refer to the `examples/training/custom` directory for more training details.
```shell
python export/ls_transformer_export.py
```
7. LightSeq Transformer using int8 quantization

Export LightSeq fp16/fp32 Transformer models to int8 protobuf format, and then using int8 quantization to speedup inference. Refer to the `examples/training/custom` directory for more training details. Note that in this example, we do not need to finetune the models using fake-quantization.
```shell
python export/ls_transformer_ptq_export.py
```
**You can compare the speeds between fp16 and int8 inference using above 6th and 7th examples.**

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
