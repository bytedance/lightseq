import time
import os
import numpy as np
import tensorflow as tf

import pyseqlib

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

batch_size = 16
src_seq_len = 64
fake_inputs = {
    "src_padding": tf.zeros([batch_size, src_seq_len], dtype=tf.float16),
    "src": tf.random.uniform([batch_size, src_seq_len], maxval=50000, dtype=tf.int64),
    "key": tf.random.uniform([batch_size, src_seq_len], maxval=50000, dtype=tf.int64),
    # "key_padding": tf.zeros([batch_size, src_seq_len], dtype=tf.float16)* model._meta["pad_id"],
    "val": tf.random.uniform([batch_size, src_seq_len], maxval=50000, dtype=tf.int64),
    # "val_padding": tf.zeros([batch_size, src_seq_len], dtype=tf.float16)* model._meta["pad_id"],
    "kv_padding": tf.zeros([batch_size, src_seq_len], dtype=tf.float16),
    "trg_input": tf.random.uniform(
        [batch_size, src_seq_len], maxval=50000, dtype=tf.int64
    ),
}
model = tf.saved_model.load("/data00/home/xiongying.taka/rewriting_model/1")
test_enc_out, _ = model.serve_encoder(fake_inputs)
test_enc_out = test_enc_out.numpy()
decoder = pyseqlib.TransformerDecoder(
    "/data00/home/xiongying.taka/projects/bytedseq/transformer_rewriting.pb", 32
)
# test_enc_out = np.load("/data00/home/xiongying.taka/test_vtm.npy")
res = decoder.infer(test_enc_out)
print(res.shape)
start = time.time()
res = None
for _ in range(100):
    test_enc_out, _ = model.serve_encoder(fake_inputs)
    test_enc_out = test_enc_out.numpy()
    res = decoder.infer(test_enc_out)
print((time.time() - start) / 100)
