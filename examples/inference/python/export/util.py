import argparse
import tensorflow as tf
import h5py

from export.proto.transformer_pb2 import Transformer
from lightseq.training import export_pb2hdf5
from lightseq.training import export_quant_pb2hdf5


def parse_args():
    parser = argparse.ArgumentParser(description="export fairseq checkpoint", usage="")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="checkpoint_best.pt",
        help="path of fairseq checkpoint",
    )
    parser.add_argument(
        "--hdf5",
        "-hdf5",
        action="store_true",
        help="whether to store hdf5",
    )
    parser.add_argument(
        "--generation_method",
        "-g",
        type=str,
        default="beam_search",
        choices=["beam_search", "topk_greedy", "topk", "topp", "ppl"],
        help="generation method",
    )
    args = parser.parse_args()
    return args


def save_model(transformer, pb_path, hdf5_path, hdf5):
    if not hdf5:
        try:
            str_model = transformer.SerializeToString()
            print("Writing to {0}".format(pb_path))
            with tf.io.gfile.GFile(pb_path, "wb") as fout:
                fout.write(str_model)
            return pb_path
        except:
            pass

    print("Writing to {0}".format(hdf5_path))
    f = h5py.File(hdf5_path, "w")
    if isinstance(transformer, Transformer):
        export_pb2hdf5(transformer, f)
    else:
        export_quant_pb2hdf5(transformer, f)
    f.close()
    return hdf5_path
