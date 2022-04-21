# Examples of exporting models for LightSeq inference

## Switch to the current directory
```shell
cd examples/inference/python
```

## Export models
### Hugging Face
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
4. Hugging Face ViT

Export Hugging Face ViT models to hdf5 format.
```shell
python export/huggingface/hf_vit_export.py
```
### Native Fairseq
1. Native Fairseq Transformer

Export native Fairseq Transformer models to protobuf/hdf5 format. Refer to the `examples/training/fairseq` directory for more training details.
```shell
python export/fairseq/native_fs_transformer_export.py -m checkpoint_best.pt
```

2. Native Fairseq Transformer using PTQ

Export native Fairseq Transformer models using PTQ to protobuf/hdf5 format. Refer to the `examples/training/fairseq` directory for more training details.
```shell
python export/fairseq/native_fs_transformer_export.py -m checkpoint_best.pt
```

3. Native Fairseq MoE Transformer

Export Fairseq MoE models to protobuf/hdf5 format.
```shell
python export/fairseq/fs_moe_export.py
```

### Fairseq Transformer + LightSeq
1. Fairseq Transformer using LightSeq training library

Export Fairseq Transformer models training with LightSeq to protobuf/hdf5 format. Refer to the `examples/training/fairseq` directory for more training details.
```shell
python export/fairseq/ls_fs_transformer_export.py -m checkpoint_best.pt
```

2. Fairseq Transformer using LightSeq training library with PTQ

Export Fairseq Transformer models training with LightSeq to protobuf format, and then using PTQ to speedup inference. Refer to the `examples/training/fairseq` directory for more training details.
```shell
python export/fairseq/ls_fs_transformer_ptq_export.py -m checkpoint_best.pt
```

### LightSeq Transformer

1. LightSeq Transformer

Export LightSeq Transformer models to protobuf/hdf5 format. Refer to the `examples/training/custom` directory for more training details.
```shell
python export/ls_transformer_export.py
```
2. LightSeq Transformer using PTQ

Export LightSeq fp16/fp32 Transformer models to int8 protobuf format, and then using PTQ to speedup inference. Refer to the `examples/training/custom` directory for more training details. Note that in this example, we do not need to finetune the models using fake-quantization.
```shell
python export/ls_transformer_ptq_export.py
```

### Fairseq Transformer + custom Torch layers
1. Fairseq Transformer using custom Torch layers

Export Fairseq Transformer models training using custom Torch layers to protobuf/hdf5 format. Refer to the `examples/training/fairseq` directory for more training details.
```shell
python export/fairseq/ls_torch_fs_transformer_export.py -m checkpoint_best.pt
```

2. Fairseq Transformer using custom Torch layers and PTQ

Export PTQ Fairseq Transformer models training using custom Torch layers to protobuf/hdf5 format. Refer to the `examples/training/fairseq` directory for more training details.
```shell
python export/fairseq/ls_torch_fs_transformer_ptq_export.py -m checkpoint_best.pt
```

3. Quantized Fairseq Transformer using custom Torch layers

Export quantized Fairseq Transformer models training using custom Torch layers to protobuf/hdf5 format. Refer to the `examples/training/fairseq` directory for more training details.
```shell
python export/fairseq/ls_torch_fs_quant_transformer_export.py -m checkpoint_best.pt
```

## Inference using LightSeq
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
4. ViT
```shell
python test/ls_vit.py
```

5. Fairseq based models using LightSeq inference
```shell
bash test/ls_fairseq.sh --model ${model_path}
```
