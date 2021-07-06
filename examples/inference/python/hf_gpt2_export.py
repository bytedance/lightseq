import os
import h5py
import numpy as np
from collections import OrderedDict
from transformers import GPT2LMHeadModel
from utils import fill_hdf5_layer

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
    }
)


def extract_gpt_weights(
    output_file,
    model_dir,
    head_num,
    generation_method,
    topk=1,
    topp=0.75,
    # default eos_id from https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel
    eos_id=50256,
    pad_id=50257,
):
    # load var names
    encoder_state_dict = GPT2LMHeadModel.from_pretrained(model_dir).state_dict()
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

    # fill src_embedding
    fill_hdf5_layer(
        enc_var_name_list,
        encoder_state_dict,
        hdf5_file,
        "src_embedding/",
        src_emb_mapping_dict,
    )

    # save number of layers metadata
    hdf5_file.create_dataset(
        "model_conf/n_encoder_stack", data=len(enc_tensor_names), dtype="i4"
    )
    # fill in model_conf
    hdf5_file.create_dataset("model_conf/head_num", data=head_num, dtype="i4")
    hdf5_file.create_dataset("model_conf/src_padding_id", data=pad_id, dtype="i4")
    hdf5_file.create_dataset(
        "model_conf/sampling_method",
        data=np.array([ord(c) for c in generation_method]).astype(np.int8),
        dtype="i1",
    )
    hdf5_file.create_dataset("model_conf/topp", data=topp, dtype="f4")
    hdf5_file.create_dataset("model_conf/topk", data=topk, dtype="i4")
    hdf5_file.create_dataset("model_conf/eos_id", data=eos_id, dtype="i4")

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
    output_lightseq_model_name = "lightseq_gpt2_base"  # or "lightseq_gpt2_large"
    input_huggingface_gpt_model = "gpt2"  # or "gpt2-large"
    head_number = 12  # 20 for "gpt2-large"
    # generation_method should be "topk" or "topp"
    generation_method = "topk"
    topk = 1
    topp = 0.75
    # default eos_id from https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel
    eos_id = 50256
    pad_id = 50257
    extract_gpt_weights(
        output_lightseq_model_name,
        input_huggingface_gpt_model,
        head_num=head_number,  # layer number
        generation_method=generation_method,
        topk=topk,
        topp=topp,
        eos_id=eos_id,
        pad_id=pad_id,
    )
