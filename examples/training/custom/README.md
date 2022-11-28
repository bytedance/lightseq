# Building models from scratch

This repo contains an example for how to use LightSeq to build model from scratch. In this example, we train a Transformer model use LightSeq Transformer model, cross entropy layer and adam optimizer.

The source inputs of the encoder are batch of sentences and the target outputs of the decoder are their corresponding replies. We use Hugging Face tokenizer to obtain the token indexes of the sentences.

You can run the example simplely by:
```shell
python examples/training/custom/run.py
```

If it runs successfully, you will see the following output:
```text
========================TRAIN========================
TransformerEmbeddingLayer #0 bind weights and grads.
TransformerEncoderLayer #0 bind weights and grads.
TransformerEncoderLayer #1 bind weights and grads.
TransformerEncoderLayer #2 bind weights and grads.
TransformerEncoderLayer #3 bind weights and grads.
TransformerEncoderLayer #4 bind weights and grads.
TransformerEncoderLayer #5 bind weights and grads.
TransformerEmbeddingLayer #1 bind weights and grads.
TransformerDecoderLayer #0 bind weights and grads.
Decoder layer #0 allocate encdec_kv memory
TransformerDecoderLayer #1 bind weights and grads.
TransformerDecoderLayer #2 bind weights and grads.
TransformerDecoderLayer #3 bind weights and grads.
TransformerDecoderLayer #4 bind weights and grads.
TransformerDecoderLayer #5 bind weights and grads.
epoch 000: 725.560
epoch 200: 96.252
epoch 400: 15.151
epoch 600: 5.770
epoch 800: 3.212
epoch 1000: 1.748
epoch 1200: 0.930
epoch 1400: 0.457
epoch 1600: 0.366
epoch 1800: 0.299
========================TEST========================
>>>>> source text
What is the fastest library in the world?
You are so pretty!
What do you love me for?
The sparrow outside the window hovering on the telephone pole.
>>>>> target text
I guess it must be LightSeq, because ByteDance is the fastest.
Thanks very much and you are pretty too.
Love your beauty, smart, virtuous and kind.
You said all this is very summery.
>>>>> predict text
I guess it must be LightSeq, because ByteDance is the fastest.
Thanks very much and you are pretty too.
Love your beauty, smart, virtuous and kind.
You said all this is very summery.
```
