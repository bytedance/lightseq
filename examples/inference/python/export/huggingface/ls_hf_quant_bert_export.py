"""
Export LightSeq quantized BERT models to hdf5 format.
"""
import os
import h5py
from collections import OrderedDict

import torch
from lightseq.training.ops.pytorch.export_quant import (
    export_ls_quant_encoder,
    fill_quant_hdf5_layer,
    quantize,
)
from export.util import parse_args

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


"""
For the mapping dictionary: key is the value of the proto parameter,
value is a powerful expression, each && split tensor name of the matching path or expression.

The sub-pattern of the path is separated by spaces, and the expression starts with a expression_.
You can operate separately on each tensor and support multiple expressions. Multiple matching paths
and the expression will finally be concatenated on axis = -1.
"""
src_emb_mapping_dict = OrderedDict(
    {
        "norm_scale": "embeddings LayerNorm weight",
        "norm_bias": "embeddings LayerNorm bias",
        "position_embedding": "embeddings position_embeddings weight",
    }
)


def extract_bert_weights(
    output_file,
    model_dir,
    head_num,
    pad_id=0,
):
    # load var names
    state_dict = torch.load(model_dir, "cpu")
    var_name_list = list(state_dict.keys())

    # initialize output file
    print("Saving model to hdf5...")
    print("Writing to {0}".format(output_file))
    hdf5_file = h5py.File(output_file, "w")

    wte = state_dict["bert.embeddings.word_embeddings.weight"]
    emb_dim = wte.shape[1]
    layer_nums = 0
    for name in var_name_list:
        if name.endswith("para"):
            layer_nums += 1

    export_ls_quant_encoder(hdf5_file, state_dict, emb_dim, emb_dim * 4, False)

    # fill src_embedding - except for position embedding
    fill_quant_hdf5_layer(
        var_name_list,
        state_dict,
        hdf5_file,
        "src_embedding/",
        src_emb_mapping_dict,
        layer_nums,
    )

    # handling token_embeddings for BERT
    token_embedding = (
        state_dict["bert.embeddings.word_embeddings.weight"]
        + state_dict["bert.embeddings.token_type_embeddings.weight"][0]
    )
    token_embedding = quantize(
        token_embedding.numpy(),
        127,
        state_dict["bert.embeddings.emb_quant.clip.clip_value_max"].numpy(),
    )
    print(f"processed token_embedding, shape: {token_embedding.shape}")
    hdf5_file.create_dataset(
        "src_embedding/token_embedding", data=token_embedding, dtype="uint8"
    )
    hdf5_file.create_dataset(
        "src_embedding/emb_clip_max",
        data=state_dict["bert.embeddings.emb_quant.clip.clip_value_max"],
    )

    # save number of layers metadata
    hdf5_file.create_dataset("model_conf/n_encoder_stack", data=layer_nums, dtype="i4")
    # fill in model_conf
    hdf5_file.create_dataset("model_conf/head_num", data=head_num, dtype="i4")
    hdf5_file.create_dataset("model_conf/src_padding_id", data=pad_id, dtype="i4")
    hdf5_file.create_dataset("model_conf/is_post_ln", data=True, dtype="?")
    hdf5_file.create_dataset("model_conf/use_gelu", data=True, dtype="?")

    # Move layernorm weights to match layernorm implementation in lightseq
    tmp_scale, tmp_bias = (
        hdf5_file["src_embedding/norm_scale"][()],
        hdf5_file["src_embedding/norm_bias"][()],
    )
    for layer_id in range(layer_nums):
        new_tmp_scale = hdf5_file[f"encoder_stack/{layer_id}/multihead_norm_scale"][()]
        new_tmp_bias = hdf5_file[f"encoder_stack/{layer_id}/multihead_norm_bias"][()]
        hdf5_file[f"encoder_stack/{layer_id}/multihead_norm_scale"][()] = tmp_scale
        hdf5_file[f"encoder_stack/{layer_id}/multihead_norm_bias"][()] = tmp_bias
        tmp_scale, tmp_bias = new_tmp_scale, new_tmp_bias

        new_tmp_scale = hdf5_file[f"encoder_stack/{layer_id}/ffn_norm_scale"][()]
        new_tmp_bias = hdf5_file[f"encoder_stack/{layer_id}/ffn_norm_bias"][()]
        hdf5_file[f"encoder_stack/{layer_id}/ffn_norm_scale"][()] = tmp_scale
        hdf5_file[f"encoder_stack/{layer_id}/ffn_norm_bias"][()] = tmp_bias
        tmp_scale, tmp_bias = new_tmp_scale, new_tmp_bias
    hdf5_file["src_embedding/norm_scale"][()] = tmp_scale
    hdf5_file["src_embedding/norm_bias"][()] = tmp_bias

    hdf5_file.close()
    # read-in again to double check
    hdf5_file = h5py.File(output_file, "r")

    def _print_pair(key, value):
        if key == "sampling_method":
            value = "".join(map(chr, value[()]))
        else:
            value = value[()]
        print(f"{key}: {value}")

    list(map(lambda x: _print_pair(*x), hdf5_file["model_conf"].items()))


if __name__ == "__main__":
    args = parse_args()
    model_name = ".".join(args.model.split(".")[:-1])
    hdf5_path = f"{model_name}.hdf5"

    head_number = 12
    pad_id = 0
    extract_bert_weights(
        hdf5_path,
        args.model,
        head_num=head_number,
        pad_id=pad_id,
    )
