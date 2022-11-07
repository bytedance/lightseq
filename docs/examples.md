## Cpp Examples
We provide multiple cpp examples of LightSeq inference.

First you should use the training examples in the following to train a model, and then export it to protobuf or HDF5 format.

Then use the cpp examples to infer the models:
1. Uncomment the `add_subdirectory(examples/inference/cpp)` in the [CMakeLists.txt](../CMakeLists.txt).
2. Build the LightSeq. Refer to [build.md](./build.md) for more details.
3. Switch to `build/temp.linux-xxx/examples/inference/cpp`, and then run `sudo make` to compile the cpp example.
4. Run the cpp examples by `./xxx_example MODEL_PATH`.

## Python Examples
We provide a series of Python examples to show how to use LightSeq to do model training and inference.

### Train the models
Currently, LightSeq supports training from [Fairseq](../examples/training/fairseq/README.md), [Hugging Face](../examples/training/huggingface/README.md), [DeepSpeed](../examples/training/deepspeed/README.md) and [from scratch](../examples/training/custom/README.md). For more training details, please refer to the respective README.

### Export and infer the models
First export the models training by Fairseq, Hugging Face or LightSeq to protobuf or HDF5 format. Then test the results and speeds using the testing scripts.

Refer to [here](../examples/inference/python/README.md) for more details.

## Depoly using Tritonbackend
Refer to [here](../examples/triton_backend/README.md) for more details.