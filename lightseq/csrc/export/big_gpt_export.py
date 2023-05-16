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
'transformer.wte.weight', 'transformer.wpe.weight', 'transformer.h.0.ln_1.weight', 'transformer.h.0.ln_1.bias', 'transformer.h.0.attn.bias', 'transformer.h.0.attn.masked_bias', 'transformer.h.0.attn.c_attn.weight', 'transformer.h.0.attn.c_attn.bias', 'transformer.h.0.attn.c_proj.weight', 'transformer.h.0.attn.c_proj.bias', 'transformer.h.0.ln_2.weight', 'transformer.h.0.ln_2.bias', 'transformer.h.0.mlp.c_fc.weight', 'transformer.h.0.mlp.c_fc.bias', 'transformer.h.0.mlp.c_proj.weight', 'transformer.h.0.mlp.c_proj.bias'
'transformer.ln_f.weight', 'transformer.ln_f.bias', 'transformer.pre_token_proj.weight', 'transformer.pre_token_proj.bias', 'transformer.post_token_proj.weight', 'transformer.post_token_proj.bias', 'lm_head.weight'
"""

dec_layer_mapping_dict = OrderedDict(
    {
        "multihead_norm_scale": "ln_1 weight",
        "multihead_norm_bias": "ln_1 bias",
        # GPT2's Conv1D don't need transpose
        # https://github.com/huggingface/transformers/blob/9ec0f01b6c3aff4636869aee735859fb6f89aa98/src/transformers/modeling_utils.py#L1400
        "multihead_project_kernel_qkv": "attn c_attn weight",
        "multihead_project_bias_qkv": "attn c_attn bias",
        "multihead_project_kernel_output": "attn c_proj weight",
        "multihead_project_bias_output": "attn c_proj bias",
        "ffn_norm_scale": "ln_2 weight",
        "ffn_norm_bias": "ln_2 bias",
        "ffn_first_kernel": "mlp c_fc weight",
        "ffn_first_bias": "mlp c_fc bias",
        "ffn_second_kernel": "mlp c_proj weight",
        "ffn_second_bias": "mlp c_proj bias",
    }
)

src_emb_mapping_dict = OrderedDict(
    {
        "norm_scale": "ln_f weight",
        "norm_bias": "ln_f bias",
        "token_embedding": "wte",
        "position_embedding": "wpe",
        # manually process position_embedding to customize for max_step
        "pre_token_proj_weight": "pre_token_proj weight&&expression_.transpose(0, 1)",
        "pre_token_proj_bias": "pre_token_proj bias",
        "post_token_proj_weight": "post_token_proj weight&&expression_.transpose(0, 1)",
        "post_token_proj_bias": "post_token_proj bias",
        "logits_linear_weight": "lm_head weight&&expression_.transpose(0, 1)",
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
    # print(enc_var_name_list)

    s = ['transformer.wte.weight', 'transformer.wpe.weight', 'transformer.h.0.ln_1.weight', 'transformer.h.0.ln_1.bias', 'transformer.h.0.attn.bias', 'transformer.h.0.attn.masked_bias', 'transformer.h.0.attn.c_attn.weight', 'transformer.h.0.attn.c_attn.bias', 'transformer.h.0.attn.c_proj.weight', 'transformer.h.0.attn.c_proj.bias', 'transformer.h.0.ln_2.weight', 'transformer.h.0.ln_2.bias', 'transformer.h.0.mlp.c_fc.weight', 'transformer.h.0.mlp.c_fc.bias', 'transformer.h.0.mlp.c_proj.weight', 'transformer.h.0.mlp.c_proj.bias',
'transformer.ln_f.weight', 'transformer.ln_f.bias', 'transformer.pre_token_proj.weight', 'transformer.pre_token_proj.bias', 'transformer.post_token_proj.weight', 'transformer.post_token_proj.bias', 'lm_head.weight'] 

    for k in s:
        print(k, state_dict[k].shape, state_dict[k].dtype)
    # exit(0)

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
    # for layer_id in sorted(enc_tensor_names.keys()):
    #     fill_hdf5_layer(
    #         enc_tensor_names[layer_id],
    #         state_dict,
    #         hdf5_file,
    #         f"decoder_layers/{layer_id}/",
    #         dec_layer_mapping_dict,
    #     )

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
        "model_conf/embed_size", data=arguments.embed_size, dtype="i4"
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
    output_lightseq_model_name = "_".join(["big_gpt", basename, "13b"])
    # default eos_id from https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel

    arguments.eos_id = 2  # need to set
    arguments.padding_id = 0  # need to set

    if not check_arguements(arguments):
        exit(0)

    extract_llama_weights(output_lightseq_model_name, arguments)
