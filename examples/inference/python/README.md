# Model export and LightSeq inference
This repo contains examples of exporting models (LightSeq, Fairseq based, Hugging Face, etc.) to protobuf/hdf5 format, and then use LightSeq for fast inference. For each model, we provide normal float model export, quantized model export (QAT, quantization aware training) and PTQ (post training quantization) model export.

Before doing anything, you need to switch to the current directory:
```shell
cd examples/inference/python
```

## Model export
We provide the following export examples. All Fairseq based models are trained using the scripts in [examples/training/fairseq](../../../examples/training/fairseq). The first two LightSeq Transformer models are trained using the scripts in [examples/training/custom](../../../examples/training/custom).

| Model                                | Type  | Command                                                                                               | Resource                                                                                                                             | Description                                                                                                                                              |
|--------------------------------------|-------|-------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| LightSeq Transformer                 | Float | python export/ls_transformer_export.py -m ckpt_ls_custom.pt                                           | [link](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/lightseq/example_model/ckpt_ls_custom.pt)                            | Export LightSeq Transformer models to protobuf format.                                                                                                   |
| LightSeq Transformer + PTQ           | Int8  | python export/ls_transformer_ptq_export.py -m ckpt_ls_custom.pt                                       | [link](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/lightseq/example_model/ckpt_ls_custom.pt)                            | Export LightSeq Transformer models to int8 protobuf format using post training quantization.                                                             |
| Hugging Face BART                    | Float | python export/huggingface/hf_bart_export.py                                                           | /                                                                                                                                    | Export Hugging Face BART models to protobuf/hdf5 format.                                                                                                 |
| Hugging Face BERT                    | Float | python export/huggingface/hf_bert_export.py                                                           | /                                                                                                                                    | Export Hugging Face BERT models to hdf5 format.                                                                                                          |
| Hugging Face GPT2                    | Float | python export/huggingface/hf_gpt2_export.py                                                           | /                                                                                                                                    | Export Hugging Face GPT2 models to hdf5 format.                                                                                                          |
| Native Fairseq Transformer           | Float | python export/fairseq/native_fs_transformer_export.py -m ckpt_native_fairseq_31.06.pt                 | [link](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/lightseq/example_model/fairseq/ckpt_native_fairseq_31.06.pt)         | Export native Fairseq Transformer models to protobuf/hdf5 format.                                                                                        |
| Native Fairseq Transformer + PTQ     | Int8  | python export/fairseq/native_fs_transformer_export.py -m ckpt_native_fairseq_31.06.pt                 | [link](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/lightseq/example_model/fairseq/ckpt_native_fairseq_31.06.pt)         | Export native Fairseq Transformer models to int8 protobuf format using post training quantization.                                                       |
| Fairseq + LightSeq Transformer       | Float | python export/fairseq/ls_fs_transformer_export.py -m ckpt_ls_fairseq_31.17.pt                         | [link](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/lightseq/example_model/fairseq/ckpt_ls_fairseq_31.17.pt)             | Export Fairseq Transformer models training with LightSeq modules to protobuf/hdf5 format.                                                                |
| Fairseq + LightSeq Transformer + PTQ | Int8  | python export/fairseq/ls_fs_transformer_ptq_export.py -m ckpt_ls_fairseq_31.17.pt                     | [link](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/lightseq/example_model/fairseq/ckpt_ls_fairseq_31.17.pt)             | Export Fairseq Transformer models training with LightSeq modules to int8 protobuf format using post training quantization.                               |
| Fairseq + custom Torch layer         | Float | python export/fairseq/ls_torch_fs_transformer_export.py -m ckpt_ls_torch_fairseq_31.16.pt             | [link](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/lightseq/example_model/fairseq/ckpt_ls_torch_fairseq_31.16.pt)       | Export Fairseq Transformer models training with custom Torch layers and other LightSeq modules to protobuf format.                                       |
| Fairseq + custom Torch layer + PTQ   | Int8  | python export/fairseq/ls_torch_fs_transformer_ptq_export.py -m ckpt_ls_torch_fairseq_31.16.pt         | [link](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/lightseq/example_model/fairseq/ckpt_ls_torch_fairseq_31.16.pt)       | Export Fairseq Transformer models training with custom Torch layers and other LightSeq modules to int8 protobuf format using post training quantization. |
| Fairseq + custom Torch layer + QAT   | Int8  | python export/fairseq/ls_torch_fs_quant_transformer_export.py -m ckpt_ls_torch_fairseq_quant_31.09.pt | [link](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/lightseq/example_model/fairseq/ckpt_ls_torch_fairseq_quant_31.09.pt) | Export quantized Fairseq Transformer models training with custom Torch layers and other LightSeq modules to int8 protobuf format.                        |
| Native Fairseq MoE Transformer       | Float | python export/fairseq/native_fs_moe_transformer_export.py                                             | /                                                                                                                                    | Export Fairseq MoE Transformer models to protobuf/hdf5 format.                                                                                           |

## LightSeq inference
### Hugging Face models
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

### Fairseq based models
After exporting the Fairseq based models to protobuf/hdf5 format using above scripts, we can use the following script for fast LightSeq inference on wmt14 en2de dateset, compatible with fp16 and int8 models:
```shell
bash test/ls_fairseq.sh --model ${model_path}
```
