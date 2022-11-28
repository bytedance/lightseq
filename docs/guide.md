# A Guide of LightSeq Training and Inference
## Introduction
This document mainly introduces the detailed process of LightSeq training and inference. In short, the process can be divided into the following three steps:
1. Train models integrated with LightSeq training modules, and save the checkpoints.
2. Export the checkpoints to protobuf/hdf5 format for futher inference.
3. Load the protobuf/hdf5 format models into LightSeq inference engine.

## Training
LightSeq provide efficient embedding, transformer encoder/decoder layer, cross entropy and adam training modules. It also provides training examples of [Fairseq](https://github.com/pytorch/fairseq), [Hugging Face](https://github.com/huggingface/transformers), [DeepSpeed](https://github.com/microsoft/DeepSpeed), [NeurST](https://github.com/bytedance/neurst) and custom integration. Refer to [training examples](../examples/training) for more details of code implementations.

### Custom integration
First, import all modules which may be used:
```python
from lightseq.training import (
    LSTransformer,
    LSTransformerEmbeddingLayer,
    LSTransformerEncoderLayer,
    LSTransformerDecoderLayer,
    LSCrossEntropyLayer,
    LSAdam,
)
```

Take transformer encoder layer as an example, the creation can be divided into two steps:
1. Create a configuration using `LSTransformerEncoderLayer.get_config`.
2. Create a LightSeq Transformer encoder layer using `LSTransformerEncoderLayer` class, initialized with the configuration.

A detailed implementation is as follows, in which `model` specifies the model architecture (`transformer-base`, `transformer-big`, `bert-base` and `bert-big`):
```python
config = LSTransformerEncoderLayer.get_config(
    model="bert-base",
    max_batch_tokens=4096,
    max_seq_len=512,
    fp16=True,
    local_rank=0,
)
layer = LSTransformerEncoderLayer(config)
```

You can also use the other model architectures by specify all parameters:
```python
config = LSTransformerEncoderLayer.get_config(
    max_batch_tokens=4096,
    max_seq_len=512,
    hidden_size=1024,
    intermediate_size=4096,
    nhead=16,
    attn_prob_dropout_ratio=0.1,
    activation_dropout_ratio=0.1,
    hidden_dropout_ratio=0.1,
    pre_layer_norm=False,
    activation_fn="gelu",
    fp16=True,
    local_rank=0,
)
layer = LSTransformerEncoderLayer(config)
```

In addition to encoder layers, the other modules can be created using similar methods, and then be trained as normal PyTorch models.

LightSeq also provides complete Transformer interface for usage:
```python
config = LSTransformer.get_config(
    model="transformer-base",
    max_batch_tokens=4096,
    max_seq_len=512,
    vocab_size=32000,
    padding_idx=0,
    num_encoder_layer=6,
    num_decoder_layer=6,
    fp16=True,
    local_rank=0,
)
model = LSTransformer(config)
```

More details are available [here](../examples/training/custom), and you can run `python run.py` to see the effect.

### Hugging Face
Hugging Face creates the models by `AutoModel.from_pretrained`. To use LightSeq to accelerate the training process, all encoder/decoder layers in original models must be replaced by corresponding LightSeq modules. Taking the GLUE task as an example, the replacement process have three steps:
1. Create a configuration using `LSTransformerEncoderLayer.get_config`.
2. Extract all pretrained model weights from the Hugging Face checkpoints.
3. Create a LightSeq Transformer encoder layer using `LSTransformerEncoderLayer` class, initialized with the configuration and pretrained weights.

Compared to custom integration, the second step can initialize the LightSeq modules using pretrained weights.

More details are available [here](../examples/training/huggingface), and you can run `sh run_glue.sh` and `sh run_ner.sh` to see the effects on GLUE and NER tasks respectively.

**NOTE:** The fine-tuning of Hugging Face BERT is unstable and may not converge. You can try to modify the `--seed` parameter in the running script to fix it.

### Fairseq
LightSeq integrates all the above modules into Fairseq. After installing the LightSeq library, you can directly use `lightseq-train` instead of `fairseq-train` to start the Fairseq training using LightSeq. The detailed usage is available [here](../examples/training/fairseq).

### DeepSpeed
Similar to Fairseq, you can use `deepspeed` to start the Fairseq training and use `--user-dir` to specify the Fairseq modules using LightSeq. More details are available [here](../examples/training/deepspeed)

## Inference
### Export
After the model is trained, you can directly load the saved checkpoint to fine-tune or infer. But this will call the inference part of the training models, which is actually the forward propagation. It needs to frequently switch between Python and C++ code, and a lot of variables for backpropagation are calculated, which is not needed. Therefore, the speed is slower than LightSeq inference engine.

To use the LightSeq inference engine, you must export the checkpoints to protobuf or hdf5 format. LightSeq defines the proto of Transformer models, and the details can be available [here](inference/export_model.md).

The export process can be divided into two steps:
1. Create an object of Transformer proto or hdf5 file.
2. Extract all weights in the checkpoints and assign them to the object.

If you use LightSeq modules for training, you can directly call the export interface provided by LightSeq. First, the following functions must be imported:
```python
from lightseq.training import (
    export_ls_config,
    export_ls_embedding,
    export_ls_encoder,
    export_ls_decoder,
)
```

These functions can export the configuration, embedding, encoder and decoder weights into the pre-defined proto. Other weights (e.g., decoder output projection) can be exported manually.

LightSeq provides export examples of native Hugging Face BERT/BART/GPT2, Fairseq trained with LightSeq and LightSeq Transformer. All codes are available [here](../examples/inference/python/export).

#### Fairseq
The main code is as follows (some parameters are omitted). Complete code is available [here](../examples/inference/python/export/ls_fs_transformer_export.py).
```python
model = Transformer()
encoder_state_dict, decoder_state_dict = _extract_weight(state_dict)
export_ls_embedding(model, encoder_state_dict, is_encoder=True)
export_ls_embedding(model, encoder_state_dict, is_encoder=False)
export_ls_encoder(model, encoder_state_dict)
export_ls_decoder(model, decoder_state_dict)
export_fs_weights(model, state_dict)
export_ls_config(model)
```

First, you need to divide the state dict into two parts of encoder and decoder, for that LightSeq can not parse the outermost weight names of user-defined models. Second, export the weights of embedding, encoder and decoder using the LightSeq interfaces. Third, export the left weights which do not use the LightSeq modules, such as decoder output projection. Finally, export the configuration of models, such as beam size, generation method, etc.

The above functions export the checkpoints to protobuf by default. Specify `save_pb=False` to export to hdf5 files. You can use the [Fairseq training example](../examples/training/fairseq) to obtain the trained checkpoints.

#### Hugging Face
LightSeq provides three examples of exporting native Hugging Face models ([BERT](../examples/inference/python/export/hf_bert_export.py), [BART](../examples/inference/python/export/hf_bart_export.py) and [GPT2](../examples/inference/python/export/hf_gpt2_export.py)). Because these native models did not use LightSeq modules to pretrain, the users must manually make the export rules.

#### LightSeq Transformer
LightSeq provide an example of exporting its own Transformer module, which is similar to Fairseq models export. You can use the [custom training example](../examples/training/custom) to obtain the trained checkpoints. This export example can also compare the results and speeds of forward propagation in training library, inference library loading both protobuf and hdf5 files. The results show that the inference library is faster than the forward propagation of training library by about 2x.

#### Custom models
LightSeq can not parse the parameter names of custom models which do not use LightSeq modules. Therefore you must make the export rules to extract the weights and assign them to corresponding positions in the protobuf or hdf5 files.

For example, suppose that the name of the layer norm weight in the last part of the encoder is `encoder.layer_norm.weight`, you can use the following code to export it:
```python
transformer = Transformer()
enc_norm_w = state_dict["encoder.layer_norm.weight"].flatten().tolist()
transformer.src_embedding.norm_scale[:] = enc_norm_w
```

The other weights are exported in similar ways. You can find all definitions of the proto [here](../lightseq/inference/proto).

### Inference in three lines of codes!
After exporting to the protobuf or hdf5 files, LightSeq can easily infer using only three lines of codes:
```python
import lightseq.inference as lsi
model = lsi.Transformer("transformer.pb", 8)
output = model.infer([[1, 2, 3], [4, 5, 6]])
```

The path of model files and the maximal batch size must be specified, and the input array must be 2-dim.

LightSeq provides three inference examples of Hugging Face BERT/BART/GPT2. The codes can be available [here](../examples/inference/python/test).
