# Build models from scratch
This repo contains an example for how to use LightSeq to build model from scratch. In this example, we train a Transformer model using LightSeq Transformer model, cross entropy layer and adam optimizer.

The source inputs of the encoder are batch of sentences and the target outputs of the decoder are their corresponding replies. We use Hugging Face tokenizer to obtain the token indexes of the sentences.

You can run the example simplely by:
```shell
python3 examples/training/custom/run.sh
```

(Optional) You can also train the model using int8 mixed-precision:
```shell
python3 examples/training/custom/run_quant.sh
```
