import time

import numpy as np

import pyseqlib

decoder = pyseqlib.TransformerDecoder("/data00/home/xiongying.taka/vtm_decoder.pb", 64)
test_enc_out = np.load("/data00/home/xiongying.taka/test_vtm.npy")
print(test_enc_out.shape)
start = time.time()
res = None
for _ in range(1):
    res = decoder.infer(test_enc_out)
print(res)
print((time.time() - start) / 1)
assert res.shape == (64, 20)
