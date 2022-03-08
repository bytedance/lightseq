import random

import numpy as np
import tensorflow as tf
import torch
from lightseq.training.pytorch_quantization import tensor_quant

test_input = (
    (
        np.random.rand(
            int(1e6),
        )
        - 0.5
    )
    * 100
).astype(np.float16)


clip_max = random.random() * 100

# normal quant test
def custom_quant(input, scale, clip_min, clip_max):
    res = input.float() / scale
    res = (res + 0.5).floor()
    # res = res.round()
    res = res.clip(clip_min, clip_max)
    res = res * scale
    return res.to(input)


tf_res = tf.quantization.fake_quant_with_min_max_vars(
    tf.constant(test_input, dtype=tf.float32), -clip_max, clip_max, 8, True
)
tf_res = tf.cast(tf_res, tf.float16)

torch_res = torch.fake_quantize_per_tensor_affine(
    input=torch.tensor(test_input),
    scale=torch.tensor(clip_max / 127, dtype=torch.float32),
    zero_point=torch.tensor(0, dtype=torch.int64),
    quant_min=-127,
    quant_max=127,
)

custom_res = custom_quant(
    torch.tensor(test_input, dtype=torch.half), clip_max / 127, -127, 127
)

nv_res = tensor_quant.fake_tensor_quant(
    torch.tensor(test_input, dtype=torch.half),
    torch.tensor(clip_max, dtype=torch.float),
)

# tf and torch's results are possible to be different because they have different rounding method
np.testing.assert_allclose(tf_res.numpy(), torch_res.numpy(), verbose=True)
np.testing.assert_allclose(custom_res.numpy(), nv_res.numpy())
np.testing.assert_allclose(custom_res.numpy(), tf_res.numpy(), verbose=True)


# activation quant test

relu_mask = test_input < 0
test_input[relu_mask] = 0
tf_res = tf.quantization.fake_quant_with_min_max_vars(
    tf.constant(test_input, dtype=tf.float32), 0, clip_max, 8, True
)
tf_res = tf.cast(tf_res, tf.float16)


torch_res = torch.fake_quantize_per_tensor_affine(
    input=torch.tensor(test_input, dtype=torch.float32),
    scale=torch.tensor(clip_max / 254, dtype=torch.float32),
    zero_point=torch.tensor(0, dtype=torch.int64),
    quant_min=0,
    quant_max=254,
).half()

custom_res = custom_quant(
    torch.tensor(test_input, dtype=torch.half), clip_max / 254, 0, 254
)

nv_res = tensor_quant.fake_tensor_quant(
    torch.tensor(test_input, dtype=torch.half),
    torch.tensor(clip_max, dtype=torch.float),
    8,
    True,
    True,
)

np.testing.assert_allclose(tf_res.numpy(), torch_res.numpy(), verbose=True)
# np.testing.assert_allclose(custom_res.numpy(), torch_res.numpy(), verbose=True)
np.testing.assert_allclose(custom_res.numpy(), nv_res.numpy(), verbose=True)
np.testing.assert_allclose(custom_res.numpy(), tf_res.numpy(), verbose=True)
