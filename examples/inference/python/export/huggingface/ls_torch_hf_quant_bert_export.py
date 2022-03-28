"""
Export Hugging Face BERT models to hdf5 format.
"""
import os
import h5py
from collections import OrderedDict
import numpy as np

import torch
from lightseq.training.ops.pytorch.export import apply_rule
from lightseq.training.ops.pytorch.export_quant import quantize
from export.fairseq.util import parse_args

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


"""
For the mapping dictionary: key is the value of the proto parameter,
value is a powerful expression, each && split tensor name of the matching path or expression.

The sub-pattern of the path is separated by spaces, and the expression starts with a expression_.
You can operate separately on each tensor and support multiple expressions. Multiple matching paths
and the expression will finally be concatenated on axis = -1.
"""
enc_layer_mapping_dict = OrderedDict(
    {
        # BERT is post_layernorm
        "multihead_norm_scale": "self_attn_layer_norm weight",
        "multihead_norm_bias": "self_attn_layer_norm bias",
        "multihead_project_kernel_qkv": "self_attn qkv_proj weight&&expression_.transpose(0, 1)",
        "multihead_project_bias_qkv": "self_attn qkv_proj bias",
        "multihead_project_kernel_output": "self_attn out_proj weight&&expression_.transpose(0, 1)",
        "multihead_project_bias_output": "self_attn out_proj bias",
        "ffn_norm_scale": "final_layer_norm weight",
        "ffn_norm_bias": "final_layer_norm bias",
        "ffn_first_kernel": "fc1 weight&&expression_.transpose(0, 1)",
        "ffn_first_bias": "fc1 bias",
        "ffn_second_kernel": "fc2 weight&&expression_.transpose(0, 1)",
        "ffn_second_bias": "fc2 bias",
        # weight_clip_max
        "multihead_project_kernel_qkv_clip_max": "self_attn qkv_proj weight_quant clip_value_max",
        "multihead_project_kernel_output_clip_max": "self_attn out_proj weight_quant clip_value_max",
        "ffn_first_kernel_clip_max": "fc1 weight_quant clip_value_max",
        "ffn_second_kernel_clip_max": "fc2 weight_quant clip_value_max",
        # act_clip_max
        "multihead_ln_clip_max": "self_attn qkv_proj input_quant clip_value_max",
        "multihead_project_output_clip_max": "self_attn out_proj input_quant clip_value_max",
        "ffn_ln_clip_max": "fc1 input_quant clip_value_max",
        "ffn_first_act_clip_max": "fc2 input_quant clip_value_max",
        "multihead_qkv_dense_clip_max": "self_attn qkv_proj output_quant clip_value_max",
        "multihead_output_dense_clip_max": "self_attn out_proj output_quant clip_value_max",
        "ffn_first_output_clip_max": "fc1 output_quant clip_value_max",
    }
)

src_emb_mapping_dict = OrderedDict(
    {
        "norm_scale": "embeddings LayerNorm weight",
        "norm_bias": "embeddings LayerNorm bias",
        "position_embedding": "embeddings position_embeddings weight",
    }
)


def fill_quant_hdf5_layer(
    tensor_names, state_dict, hdf5_file, hdf5_dataset_prefix, mapping_dict
):
    for proto_name, ckpt_rule in mapping_dict.items():
        target_tensor = apply_rule(proto_name, ckpt_rule, tensor_names, state_dict)
        if proto_name.endswith("_clip_max"):
            hdf5_file.create_dataset(
                hdf5_dataset_prefix + proto_name, data=float(target_tensor[0])
            )
        else:
            hdf5_file.create_dataset(
                hdf5_dataset_prefix + proto_name,
                data=target_tensor,
            )


def extract_bert_weights(
    output_file,
    model_dir,
    head_num,
    pad_id=0,
    max_step=50,
):
    # load var names
    state_dict = torch.load(model_dir, "cpu")

    var_name_list = list(state_dict.keys())

    for name in var_name_list:
        if name.endswith("weight_quant.clip.clip_value_max"):
            state_dict[name[:-26]] = torch.Tensor(
                quantize(state_dict[name[:-26]].numpy(), 127, state_dict[name].numpy())
            ).to(torch.uint8)

    # initialize output file
    print("Saving model to hdf5...")
    print("Writing to {0}".format(output_file))
    hdf5_file = h5py.File(output_file, "w")

    # fill each encoder layer's params
    enc_tensor_names = {}
    for name in var_name_list:
        name_split = name.split(".")
        if len(name_split) <= 3 or not name_split[3].isdigit():
            continue
        layer_id = int(name_split[3])
        enc_tensor_names.setdefault(layer_id, []).append(name)

    # fill encoder_stack
    for layer_id in sorted(enc_tensor_names.keys()):
        fill_quant_hdf5_layer(
            enc_tensor_names[layer_id],
            state_dict,
            hdf5_file,
            f"encoder_stack/{layer_id}/",
            enc_layer_mapping_dict,
        )

    # fill src_embedding - except for position embedding
    fill_quant_hdf5_layer(
        var_name_list,
        state_dict,
        hdf5_file,
        "src_embedding/",
        src_emb_mapping_dict,
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
    hdf5_file.create_dataset(
        "model_conf/n_encoder_stack", data=len(enc_tensor_names), dtype="i4"
    )
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
    for layer_id in sorted(enc_tensor_names.keys()):
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
    max_step = 50
    extract_bert_weights(
        hdf5_path,
        args.model,
        head_num=head_number,
        pad_id=pad_id,
        max_step=max_step,
    )
