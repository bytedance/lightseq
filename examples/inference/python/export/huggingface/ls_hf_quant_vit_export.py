"""
Export Hugging Face ViT models to hdf5 format.
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
        "conv_weight": "embeddings patch_embeddings projection weight",
        "conv_bias": "embeddings patch_embeddings projection bias",
        "position_embedding": "embeddings position_embeddings",
        "cls_embedding": "embeddings cls_token",
        "norm_scale": "layernorm weight",
        "norm_bias": "layernorm bias",
    }
)


def extract_vit_weights(
    output_file,
    model_dir,
    head_num,
    image_size,
    patch_size,
):
    # load var names
    encoder_state_dict = torch.load(model_dir, "cpu")
    enc_var_name_list = list(encoder_state_dict.keys())

    # initialize output file
    print("Saving model to hdf5...")
    print("Writing to {0}".format(output_file))
    hdf5_file = h5py.File(output_file, "w")

    ln = encoder_state_dict["vit.layernorm.weight"]
    emb_dim = ln.shape[0]
    layer_nums = 0
    for name in enc_var_name_list:
        if name.endswith("para"):
            layer_nums += 1

    export_ls_quant_encoder(hdf5_file, encoder_state_dict, emb_dim, emb_dim * 4, False)

    # fill src_embedding
    fill_quant_hdf5_layer(
        enc_var_name_list,
        encoder_state_dict,
        hdf5_file,
        "src_embedding/",
        src_emb_mapping_dict,
        layer_nums,
    )

    # save number of layers metadata
    hdf5_file.create_dataset("model_conf/n_encoder_stack", data=layer_nums, dtype="i4")
    # fill in model_conf
    hdf5_file.create_dataset("model_conf/head_num", data=head_num, dtype="i4")
    hdf5_file.create_dataset("model_conf/use_gelu", data=True, dtype="?")
    hdf5_file.create_dataset("model_conf/is_post_ln", data=False, dtype="?")
    hdf5_file.create_dataset("model_conf/image_size", data=image_size, dtype="i4")
    hdf5_file.create_dataset("model_conf/patch_size", data=patch_size, dtype="i4")

    hdf5_file.close()
    # read-in again to double check
    hdf5_file = h5py.File(output_file, "r")

    def _print_pair(key, value):
        value = value[()]
        print(f"{key}: {value}")

    list(map(lambda x: _print_pair(*x), hdf5_file["model_conf"].items()))


if __name__ == "__main__":
    args = parse_args()
    model_name = ".".join(args.model.split(".")[:-1])
    hdf5_path = f"{model_name}.hdf5"

    head_number = 12
    image_size = 224
    patch_size = 16

    extract_vit_weights(
        hdf5_path,
        args.model,
        head_number,
        image_size,
        patch_size,
    )
