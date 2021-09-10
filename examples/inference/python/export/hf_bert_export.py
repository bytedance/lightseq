"""
Export Hugging Face BERT models to hdf5 format.
"""
import os
import h5py
import numpy as np
from collections import OrderedDict
from transformers import BertModel
from lightseq.training.ops.pytorch.export import fill_hdf5_layer

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
        # NOTE: add an additional "final" at the beginning for some weight
        # to distinguish them from "attention output *"
        "multihead_norm_scale": "attention output LayerNorm weight",
        "multihead_norm_bias": "attention output LayerNorm bias",
        "multihead_project_kernel_qkv": "attention self query weight&&attention self key weight&&attention self value weight&&expression_.transpose(0, 1)",
        "multihead_project_bias_qkv": "attention self query bias&&attention self key bias&&attention self value bias",
        "multihead_project_kernel_output": "attention output dense weight&&expression_.transpose(0, 1)",
        "multihead_project_bias_output": "attention output dense bias",
        "ffn_norm_scale": "final output LayerNorm weight",
        "ffn_norm_bias": "final output LayerNorm bias",
        "ffn_first_kernel": "intermediate dense weight&&expression_.transpose(0, 1)",
        "ffn_first_bias": "intermediate dense bias",
        "ffn_second_kernel": "final output dense weight&&expression_.transpose(0, 1)",
        "ffn_second_bias": "final output dense bias",
    }
)

src_emb_mapping_dict = OrderedDict(
    {
        "norm_scale": "embeddings LayerNorm weight",
        "norm_bias": "embeddings LayerNorm bias",
        "position_embedding": "embeddings position_embeddings weight",
        # manually process token_embedding due to "token_type_embeddings"
        # "token_embedding": "embeddings word_embeddings weight",
    }
)


def extract_bert_weights(
    output_file,
    model_dir,
    head_num,
    pad_id=0,
    max_step=50,
):
    # load var names
    encoder_state_dict = BertModel.from_pretrained(model_dir).state_dict()

    # Insert additional "final" to some weight to prevent ambiguous match
    def _insert_final(key):
        l = key.split(".")
        l.insert(3, "final")
        return ".".join(l)

    encoder_state_dict = OrderedDict(
        [
            (_insert_final(k), v)
            if len(k.split(".")) > 3 and k.split(".")[3] == "output"
            else (k, v)
            for k, v in encoder_state_dict.items()
        ]
    )

    enc_var_name_list = list(encoder_state_dict.keys())

    # initialize output file
    output_file += ".hdf5"
    print("Saving model to hdf5...")
    print("Writing to {0}".format(output_file))
    hdf5_file = h5py.File(output_file, "w")

    # fill each encoder layer's params
    enc_tensor_names = {}
    for name in enc_var_name_list:
        name_split = name.split(".")
        if len(name_split) <= 2 or not name_split[2].isdigit():
            continue
        layer_id = int(name_split[2])
        enc_tensor_names.setdefault(layer_id, []).append(name)

    # fill encoder_stack
    for layer_id in sorted(enc_tensor_names.keys()):
        fill_hdf5_layer(
            enc_tensor_names[layer_id],
            encoder_state_dict,
            hdf5_file,
            f"encoder_stack/{layer_id}/",
            enc_layer_mapping_dict,
        )

    # fill src_embedding - except for position embedding
    fill_hdf5_layer(
        enc_var_name_list,
        encoder_state_dict,
        hdf5_file,
        "src_embedding/",
        src_emb_mapping_dict,
    )

    # handling token_embeddings for BERT
    token_embedding = (
        encoder_state_dict["embeddings.word_embeddings.weight"]
        + encoder_state_dict["embeddings.token_type_embeddings.weight"][0]
    )
    print(f"processed token_embedding, shape: {token_embedding.shape}")
    token_embedding = token_embedding.flatten().tolist()
    hdf5_file.create_dataset(
        "src_embedding/token_embedding", data=token_embedding, dtype="f4"
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
    output_lightseq_model_name = "lightseq_bert_base_uncased"
    input_huggingface_bert_model = "bert-base-uncased"
    head_number = 12

    pad_id = 0
    max_step = 50
    extract_bert_weights(
        output_lightseq_model_name,
        input_huggingface_bert_model,
        head_num=head_number,
        pad_id=pad_id,
        max_step=max_step,
    )
