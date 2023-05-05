"""
Export Hugging Face GPT2 models to hdf5 format.
"""
import __init__
import os
import h5py
import numpy as np
from collections import OrderedDict
from util import parse_args, check_arguements, ModelArguements, fill_hdf5_layer
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


"""
For the mapping dictionary: key is the value of the proto parameter,
value is a powerful expression, each && split tensor name of the matching path or expression.

The sub-pattern of the path is separated by spaces, and the expression starts with a expression_.
You can operate separately on each tensor and support multiple expressions. Multiple matching paths
and the expression will finally be concatenated on axis = -1.
"""


"""
'model.layers.0.self_attn.q_proj.weight', 'model.layers.0.self_attn.k_proj.weight', 'model.layers.0.self_attn.v_proj.weight', 'model.layers.0.self_attn.o_proj.weight', 'model.layers.0.self_attn.rotary_emb.inv_freq', 'model.layers.0.mlp.gate_proj.weight', 'model.layers.0.mlp.down_proj.weight', 'model.layers.0.mlp.up_proj.weight', 'model.layers.0.input_layernorm.weight', 'model.layers.0.post_attention_layernorm.weight'
"""

dec_layer_mapping_dict = OrderedDict(
    {
        "attention_norm_scale": "input_layernorm weight",
        "attention_project_qkv": "self_attn q_proj weight&&self_attn k_proj weight&&self_attn v_proj weight",
        "attention_output": "self_attn o_proj weight",
        "ffn_norm_scale": "post_attention_layernorm weight",
        "gate_up_project_weight": "mlp gate_proj weight&&mlp up_proj weight",
        "down_project_weight": "mlp down_proj weight",
    }
)

src_emb_mapping_dict = OrderedDict(
    {
        "post_norm_scale": "norm weight",
        "token_embedding": "embed_tokens weight",
    }
)


def extract_llama_weights(
    output_file: str,
    arguments: ModelArguements,
):
    # load var names
    state_dict = torch.load(arguments.model_file)

    head_num = arguments.head_num
    enc_var_name_list = list(state_dict.keys())

    # initialize output file
    output_file += ".hdf5"
    print("Saving model to hdf5...")
    print("Writing to {0}".format(output_file))

    # exit(0)
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
            state_dict,
            hdf5_file,
            f"decoder_layers/{layer_id}/",
            dec_layer_mapping_dict,
        )

    # fill src_embedding - except for position embedding
    fill_hdf5_layer(
        enc_var_name_list,
        state_dict,
        hdf5_file,
        "src_embedding/",
        src_emb_mapping_dict,
    )

    # save number of layers metadata
    hdf5_file.create_dataset(
        "model_conf/hidden_size", data=arguments.hidden_size, dtype="i4"
    )
    hdf5_file.create_dataset(
        "model_conf/inner_size", data=arguments.inner_size, dtype="i4"
    )
    hdf5_file.create_dataset("model_conf/max_step", data=arguments.max_step, dtype="i4")
    hdf5_file.create_dataset("model_conf/head_num", data=arguments.head_num, dtype="i4")
    hdf5_file.create_dataset(
        "model_conf/layer_num", data=arguments.layer_num, dtype="i4"
    )
    hdf5_file.create_dataset(
        "model_conf/src_padding_id", data=arguments.padding_id, dtype="i4"
    )
    hdf5_file.create_dataset(
        "model_conf/generate_method",
        data=np.array([ord(c) for c in arguments.generation_method]).astype(np.int8),
        dtype="i1",
    )
    hdf5_file.create_dataset("model_conf/topp", data=arguments.topp, dtype="f4")
    hdf5_file.create_dataset("model_conf/topk", data=arguments.topk, dtype="i4")
    hdf5_file.create_dataset("model_conf/eos_id", data=arguments.eos_id, dtype="i4")
    hdf5_file.create_dataset(
        "model_conf/extra_decode_length", data=arguments.extra_decode_length, dtype="i4"
    )
    hdf5_file.create_dataset(
        "model_conf/src_vocab_size", data=arguments.vocab_size, dtype="i4"
    )

    hdf5_file.close()
    # read-in again to double check
    hdf5_file = h5py.File(output_file, "r")

    def _print_pair(key, value):
        if key == "generate_method":
            value = "".join(map(chr, value[()]))
        else:
            value = value[()]
        print(f"{key}: {value}")

    list(map(lambda x: _print_pair(*x), hdf5_file["model_conf"].items()))


if __name__ == "__main__":
    args = parse_args()

    arguments = ModelArguements(args)
    basename = os.path.basename(arguments.model_repo)
    output_lightseq_model_name = "_".join(["lightseq_llama", basename])
    # default eos_id from https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel

    arguments.eos_id = 2  # need to set
    arguments.padding_id = 0  # need to set

    if not check_arguements(arguments):
        exit(0)

    extract_llama_weights(output_lightseq_model_name, arguments)
