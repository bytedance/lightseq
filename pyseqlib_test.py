import time
import os
import numpy as np

# import tensorflow as tf
import pickle
import lightseq

# physical_devices = tf.config.list_physical_devices("GPU")
# try:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#     # Invalid device or cannot modify virtual devices once initialized.
#     pass

# batch_size = 16
# src_seq_len = 64
# fake_inputs = {
#     "src_padding": tf.zeros([batch_size, src_seq_len], dtype=tf.float16),
#     "src": tf.random.uniform([batch_size, src_seq_len], maxval=50000, dtype=tf.int64),
#     "key": tf.random.uniform([batch_size, src_seq_len], maxval=50000, dtype=tf.int64),
#     # "key_padding": tf.zeros([batch_size, src_seq_len], dtype=tf.float16)* model._meta["pad_id"],
#     "val": tf.random.uniform([batch_size, src_seq_len], maxval=50000, dtype=tf.int64),
#     # "val_padding": tf.zeros([batch_size, src_seq_len], dtype=tf.float16)* model._meta["pad_id"],
#     "kv_padding": tf.zeros([batch_size, src_seq_len], dtype=tf.float16),
#     "trg_input": tf.random.uniform(
#         [batch_size, src_seq_len], maxval=50000, dtype=tf.int64
#     ),
# }
# multi-encoder model
# model = tf.saved_model.load("/data00/home/xiongying.taka/rewriting_model/1")

# get encoder output
# test_enc_out, _ = model.serve_encoder(fake_inputs)
# test_enc_out = test_enc_out.numpy()


def test_big_batch_bug(res):
    right_num = 0
    false_num = 0
    for i in range(res.shape[1]):
        for j in range(res.shape[0] - 1):
            if np.allclose(res[j][i], res[j + 1][i]):
                right_num += 1
            else:
                false_num += 1

    print(right_num)
    print(false_num)


def get_rewriting_dict():
    id2token = []
    with open("rewriting_dict.txt") as f:
        for line in f:
            id2token.append(line.strip().split("\t")[0])
    id2token.extend(["[PAD]", "[SOS]", "[EOS]"])
    return id2token


def get_multilingual_dict():
    word2dict = pickle.load(open("mullin_en_w2d.pkl", "rb"))
    return {i: w for w, i in word2dict.items()}


def test_rewriting():
    test_enc_out = pickle.load(open("test_enc_out.pkl", "rb"))
    test_mask = np.zeros(test_enc_out.shape[:2])
    decoder = lightseq.TransformerDecoder("rewriting_decoder.pb", 32)
    print(test_enc_out.shape)
    res = decoder.infer(test_enc_out, test_mask)

    id2token = get_rewriting_dict()
    batch_size, beam_size, _ = res.shape
    for batch_id in range(batch_size):
        for beam_id in range(beam_size):
            print(
                "batch {0} beam {1}: ".format(batch_id, beam_id)
                + "".join(
                    id2token[i]
                    for i in res[batch_id, beam_id, :]
                    if id2token[i] != "[EOS]"
                )
            )
    for batch_id in range(batch_size):
        for beam_id in range(beam_size):
            print(
                " ".join(
                    str(i) for i in res[batch_id, beam_id, :] if id2token[i] != "[EOS]"
                )
            )


# only work for multilingual branch

# def test_multilingual_title():
#     test_enc_out = pickle.load(open("test_en_encout.pkl", "rb"))
#     test_mask = np.zeros(test_enc_out.shape[:2])
#     decoder = lightseq.TransformerDecoder("mullin_title_en_decoder.pb", 8)
#     print(test_enc_out.shape)
#     start = time.time()
#     for _ in range(10):
#         res = decoder.infer(test_enc_out, test_mask)
#     print((time.time() - start) / 10)
#     id2token = get_multilingual_dict()
#     batch_size, beam_size, _ = res.shape
#     for batch_id in range(batch_size):
#         for beam_id in range(beam_size):
#             print(
#                 "batch {0} beam {1}: ".format(batch_id, beam_id)
#                 + " ".join(
#                     id2token[i]
#                     for i in res[batch_id, beam_id, :]
#                     if id2token[i] != "</s>"
#                 )
#             )
#     # for batch_id in range(batch_size):
#     #     for beam_id in range(beam_size):
#     #         print(
#     #             " ".join(str(i) for i in res[batch_id, beam_id, :] if id2token[i] != "</s>")
#     #         )


def test_en_correction():
    test_input = np.array([[894, 213, 7, 334, 479, 2] for _ in range(4)])
    transformer = lightseq.Transformer("transformer_en_correction.pb", 8)
    start = time.time()
    for _ in range(1):
        res = transformer.infer(test_input, multiple_output=True)
    print((time.time() - start) / 1)
    print(res)
    # id2token = get_multilingual_dict()
    # batch_size, beam_size, _ = res.shape
    # for batch_id in range(batch_size):
    #     for beam_id in range(beam_size):
    #         print(
    #             "batch {0} beam {1}: ".format(batch_id, beam_id)
    #             + " ".join(
    #                 id2token[i]
    #                 for i in res[batch_id, beam_id, :]
    #                 if id2token[i] != "</s>"
    #             )
    #         )
    # for batch_id in range(batch_size):
    #     for beam_id in range(beam_size):
    #         print(
    #             " ".join(str(i) for i in res[batch_id, beam_id, :] if id2token[i] != "</s>")
    #         )


def get_q2q_dict():
    id2token = []
    with open("chinese_char_5k.txt") as f:
        for line in f:
            id2token.append(line.strip().split("\t")[0])
    id2token.extend(["[PAD]", "[SOS]", "[EOS]"])
    return id2token


def test_q2q_transformer():

    test_input = np.array([[5001, 2, 36, 13, 15, 5002]])
    transformer = lightseq.Transformer("q2q_transformer.pb", 8)
    start = time.time()
    for _ in range(1):
        res = transformer.infer(test_input,multiple_output=True)
    print((time.time() - start) / 1)
    print(res)
    res = res[0]
    id2token = get_q2q_dict()
    batch_size,beam_size, _ = res.shape
    for batch_id in range(batch_size):
        for beam_id in range(beam_size):
            print(
                "batch {0} beam {1}: ".format(batch_id, beam_id)
                + " ".join(
                    id2token[i] for i in res[batch_id,beam_id, :] if id2token[i] != "<EOS>"
                )
            )


# print(res.shape)
# start = time.time()
# res = None
# for _ in range(100):
#     test_enc_out, _ = model.serve_encoder(fake_inputs)
#     test_enc_out = test_enc_out.numpy()
#     res = decoder.infer(test_enc_out)
# print((time.time() - start) / 100)

if __name__ == "__main__":
    # test_en_correction()
    test_q2q_transformer()